"""
LoRA Downloader Service

A standalone Windows service that automatically downloads approved LoRAs from CivitAI.
Runs as an infinite loop with configurable interval, featuring:
- Async downloads with aiohttp
- Retry logic with exponential backoff
- Structured logging with file rotation
- Database-based status tracking
- HTTP endpoint for remote triggering

Usage:
    python lora_downloader_service.py

Environment Variables (from .env):
    CIVITAI_API_KEY - CivitAI API key
    LORAS_FOLDER - Path to store downloaded LoRAs
    DOWNLOAD_INTERVAL_SECONDS - Interval between download checks (default: 300)
    DOWNLOADER_HTTP_PORT - Port for remote trigger endpoint (default: 9002)
"""

import os
import sys
import asyncio
import re
import json
import hashlib
import logging
import ssl
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import aiohttp
import certifi
import psycopg_pool
from dotenv import load_dotenv
from aiohttp import web

# Windows event loop policy fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()

# Configuration from environment
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", os.environ.get("API_KEY"))
LORAS_FOLDER = os.environ.get("LORAS_FOLDER", r"D:\mobians_api\loras")
DOWNLOAD_INTERVAL_SECONDS = int(os.environ.get("DOWNLOAD_INTERVAL_SECONDS", "300"))
DOWNLOADER_HTTP_PORT = int(os.environ.get("DOWNLOADER_HTTP_PORT", "9002"))

# Backend URL + shared token used to trigger user-facing push notifications
# (e.g., "your LoRA is ready!") once a download completes. Both must be set
# for notifications to fire; otherwise the hook is a no-op.
BACKEND_INTERNAL_URL = os.environ.get("BACKEND_INTERNAL_URL", "http://localhost:9001").rstrip("/")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN")

# Database configuration
DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")
DSN = f"host={DBHOST} dbname='{DBNAME}' user={DBUSER} password={DBPASS}"

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

# CivitAI suppresses some public minor-character models from the model-detail
# endpoint even though their published version endpoint remains available.
CIVITAI_SAFE_MINOR_NSFW_LEVELS = {1, 2}  # PG and PG-13
CIVITAI_LORA_MODEL_TYPES = {"LORA", "LOCON"}

# Global state
db_pool: Optional[psycopg_pool.AsyncConnectionPool] = None
http_session: Optional[aiohttp.ClientSession] = None
is_downloading = False
trigger_event = asyncio.Event()

# Configure logging with rotation
def setup_logging():
    """Setup structured logging with file rotation."""
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "lora_downloader.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()


def create_http_ssl_context() -> ssl.SSLContext:
    """Use a current CA bundle instead of the machine default trust store."""
    return ssl.create_default_context(cafile=certifi.where())


# ============================================
# DATABASE STATUS TRACKING (using lora_suggestions)
# ============================================

async def update_suggestion_status(
    version_id: int,
    status: str,
    error_message: Optional[str] = None
):
    """Update a lora_suggestion's status and last_updated_date."""
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    UPDATE lora_suggestions 
                    SET status = %(status)s, 
                        error_message = %(error_message)s,
                        last_updated_date = NOW()
                    WHERE version_id = %(version_id)s
                """, {
                    'status': status,
                    'error_message': error_message,
                    'version_id': version_id
                })
    except Exception as e:
        logger.error(f"Failed to update suggestion status: {e}")


# ============================================
# PUSH NOTIFICATION TRIGGER
# ============================================

async def notify_lora_downloaded(
    version_id: int,
    name: Optional[str],
    version: Optional[str],
    requestor: Optional[str],
) -> None:
    """Ping the backend so it can send a Web Push to the suggestion's requestor.

    No-op when BACKEND_INTERNAL_URL or INTERNAL_API_TOKEN aren't configured,
    or when the suggestion has no requestor to target.
    """
    if not INTERNAL_API_TOKEN:
        return
    if not requestor:
        return
    if http_session is None:
        return

    url = f"{BACKEND_INTERNAL_URL}/internal/notify_lora_downloaded"
    payload = {
        "version_id": version_id,
        "name": name,
        "version": version,
        "requestor": requestor,
    }
    headers = {"X-Internal-Token": INTERNAL_API_TOKEN}

    try:
        async with http_session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.warning(
                    f"notify_lora_downloaded got {resp.status} from backend: {body[:200]}"
                )
            else:
                logger.info(f"Queued LoRA-ready notification for version_id={version_id}")
    except Exception as exc:
        logger.warning(f"notify_lora_downloaded request failed: {exc}")


# ============================================
# RETRY DECORATOR
# ============================================

async def retry_with_backoff(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Execute function with exponential backoff retry."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed: {e}")
    
    raise last_exception


# ============================================
# CIVITAI API FUNCTIONS
# ============================================

async def fetch_lora_version(version_id: str) -> Optional[Dict[str, Any]]:
    """Fetch LoRA version details from CivitAI API."""
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mobians-LoRA-Downloader/1.0",
    }
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    
    async with http_session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            logger.error(f"Failed to fetch version {version_id}: HTTP {response.status}")
            return None


def build_lora_model_fallback(
    model_id: str,
    lora_version: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build safe parent metadata from a CivitAI version response.

    CivitAI's version endpoint includes a small authoritative ``model`` object.
    This is used only after the corresponding public model endpoint returns 404.
    Treating the recovered model as minor is conservative and preserves the flag
    which caused CivitAI to suppress the affected public model records.
    """
    if not isinstance(lora_version, dict):
        return None

    try:
        expected_model_id = int(model_id)
        version_model_id = int(lora_version.get("modelId"))
        nsfw_level = int(lora_version.get("nsfwLevel"))
    except (TypeError, ValueError):
        return None

    if version_model_id != expected_model_id:
        return None
    if str(lora_version.get("status") or "").casefold() != "published":
        return None
    if nsfw_level not in CIVITAI_SAFE_MINOR_NSFW_LEVELS:
        return None

    embedded_model = lora_version.get("model")
    if not isinstance(embedded_model, dict):
        return None

    name = str(embedded_model.get("name") or "").strip()
    model_type = str(embedded_model.get("type") or "").strip().upper()
    if not name or model_type not in CIVITAI_LORA_MODEL_TYPES:
        return None
    if embedded_model.get("nsfw") is not False:
        return None

    creator = embedded_model.get("creator")
    tags = embedded_model.get("tags")
    return {
        "id": expected_model_id,
        "name": name,
        "type": model_type,
        "nsfw": False,
        "minor": True,
        "creator": creator if isinstance(creator, dict) else {},
        "description": embedded_model.get("description"),
        "tags": tags if isinstance(tags, list) else [],
    }


async def fetch_lora_model(
    model_id: str,
    lora_version: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch LoRA main model page from CivitAI API."""
    url = f"https://civitai.com/api/v1/models/{model_id}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mobians-LoRA-Downloader/1.0",
    }
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    
    async with http_session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()

        if response.status == 404:
            fallback = build_lora_model_fallback(model_id, lora_version)
            if fallback:
                logger.warning(
                    f"CivitAI model {model_id} returned 404; using its published "
                    "version metadata"
                )
                return fallback

            logger.error(
                f"CivitAI model {model_id} returned 404 and its version metadata "
                "was not an eligible safe public LoRA"
            )
            return None

        logger.error(f"Failed to fetch model {model_id}: HTTP {response.status}")
        return None


# ============================================
# FILE OPERATIONS
# ============================================

def sanitize_filename(name: str) -> str:
    """Sanitize filename for filesystem compatibility."""
    name = name.replace("'", "")
    return "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in name).strip()


def is_english(text: str) -> bool:
    """Check if text contains only English characters."""
    return re.match(r'^[a-zA-Z0-9\s\-_.,!?()\'\"]+$', text) is not None


def is_preferred_preview_media(media: Dict[str, Any]) -> bool:
    """Return True for static image media that our UI can render reliably."""
    if not isinstance(media, dict):
        return False

    url = str(media.get("url") or "").strip().lower()
    if not url:
        return False

    media_type = str(media.get("type") or "").strip().lower()
    mime_type = str(media.get("mimeType") or "").strip().lower()
    metadata_mime = str((media.get("metadata") or {}).get("mimeType") or "").strip().lower()
    combined = " ".join(part for part in [media_type, mime_type, metadata_mime] if part)

    if "video" in combined or "gif" in combined:
        return False

    if any(ext in url for ext in [".mp4", ".webm", ".mov", ".avi", ".m3u8", ".gif"]):
        return False

    if media_type == "image":
        return True

    if mime_type.startswith("image/") and mime_type != "image/gif":
        return True

    if metadata_mime.startswith("image/") and metadata_mime != "image/gif":
        return True

    return any(ext in url for ext in [".png", ".jpg", ".jpeg", ".webp", ".avif"])


def select_preferred_preview_url(images: Any) -> Optional[str]:
    if not isinstance(images, list):
        return None

    valid_items = [item for item in images if isinstance(item, dict) and item.get("url")]
    preferred = [item for item in valid_items if is_preferred_preview_media(item)]
    if preferred:
        return str(preferred[0].get("url") or "").strip() or None

    return None


async def download_file_async(url: str, file_path: str, expected_hash: Optional[str] = None) -> bool:
    """
    Download file asynchronously with streaming.
    Returns True if successful, False otherwise.
    """
    # Add API key to URL
    download_url = f"{url}?token={CIVITAI_API_KEY}"
    
    # Create directory if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        async with http_session.get(download_url) as response:
            if response.status != 200:
                logger.error(f"Download failed: HTTP {response.status}")
                return False
            
            # Stream to file
            with open(file_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
        
        # Verify file hash if provided (SHA256)
        if expected_hash:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            
            computed_hash = sha256.hexdigest().upper()
            if computed_hash != expected_hash.upper():
                logger.error(f"Hash mismatch! Expected: {expected_hash}, Got: {computed_hash}")
                os.remove(file_path)
                return False
            
            logger.info("File hash verified successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        # Clean up partial file
        if os.path.exists(file_path):
            os.remove(file_path)
        return False


# ============================================
# METADATA UPLOAD
# ============================================

async def upload_metadata(metadata: Dict[str, Any]) -> Optional[int]:
    """Insert or update LoRA metadata in database."""
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                params = {
                    'name': metadata['name'],
                    'version': metadata['version'],
                    'base_model': metadata['base_model'],
                    'download_url': metadata['download_url'],
                    'is_nsfw': metadata['is_nsfw'],
                    'is_minor': metadata['is_minor'],
                    'creator': metadata.get('creator'),
                    'description': metadata.get('description'),
                    'version_description': metadata.get('version_description'),
                    'tags': json.dumps(metadata['tags']),
                    'who_added': metadata['who_added'],
                    'status': metadata['status'],
                    'trigger_words': json.dumps(metadata['trigger_words']),
                    'hashes': json.dumps(metadata['hashes']),
                    'image_url': metadata.get('image_url'),
                    'file_path': metadata.get('file_path'),
                    'version_id': metadata['version_id']
                }

                await cur.execute("""
                UPDATE lora_metadata SET
                    base_model = %(base_model)s,
                    name = %(name)s,
                    version = %(version)s,
                    download_url = %(download_url)s,
                    is_nsfw = %(is_nsfw)s,
                    is_minor = %(is_minor)s,
                    creator = %(creator)s,
                    description = %(description)s,
                    version_description = %(version_description)s,
                    tags = %(tags)s,
                    who_added = %(who_added)s,
                    status = %(status)s,
                    trigger_words = %(trigger_words)s,
                    hashes = %(hashes)s,
                    image_url = %(image_url)s,
                    file_path = %(file_path)s
                WHERE id = (
                    SELECT id
                    FROM lora_metadata
                    WHERE version_id = %(version_id)s
                    ORDER BY id
                    LIMIT 1
                )
                RETURNING id
                """, params)

                record = await cur.fetchone()
                if not record:
                    await cur.execute("""
                    INSERT INTO lora_metadata (
                        name, version, base_model, download_url, is_nsfw, is_minor, 
                        creator, description, version_description, tags, 
                        who_added, status, trigger_words, hashes, image_url, file_path, version_id
                    ) VALUES (
                        %(name)s, %(version)s, %(base_model)s, %(download_url)s, %(is_nsfw)s, %(is_minor)s, 
                        %(creator)s, %(description)s, %(version_description)s, %(tags)s, 
                        %(who_added)s, %(status)s, %(trigger_words)s, %(hashes)s, %(image_url)s, %(file_path)s, %(version_id)s
                    )
                    RETURNING id
                    """, params)
                    record = await cur.fetchone()
                
                logger.info(f"Metadata uploaded for {metadata['name']} v{metadata['version']}")
                return record[0] if record else None
                
    except Exception as e:
        logger.error(f"Failed to upload metadata: {e}")
        raise


# ============================================
# MAIN DOWNLOAD LOGIC
# ============================================

async def process_approved_loras():
    """Process all approved LoRA suggestions."""
    global is_downloading
    
    if is_downloading:
        logger.info("Download already in progress, skipping...")
        return
    
    is_downloading = True
    
    try:
        # Get approved suggestions
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT version_id, requestor, name, version 
                    FROM lora_suggestions 
                    WHERE status = 'approved'
                    ORDER BY version_id ASC
                """)
                suggestions = await cur.fetchall()
        
        if not suggestions:
            logger.info("No approved LoRAs to download")
            is_downloading = False
            return
        
        logger.info(f"Found {len(suggestions)} approved LoRA(s) to download")
        
        for version_id, requestor, suggestion_name, suggestion_version in suggestions:
            lora_display_name = f"{suggestion_name} v{suggestion_version}"
            
            try:
                # Mark as downloading
                await update_suggestion_status(version_id, "downloading")
                logger.info(f"Processing: {lora_display_name} (version_id: {version_id})")
                
                # Fetch version details with retry
                lora_version = await retry_with_backoff(fetch_lora_version, str(version_id))
                if not lora_version:
                    raise Exception(f"Failed to fetch version details for {version_id}")
                
                # Fetch main model page
                model_id = lora_version.get('modelId')
                main_page = await retry_with_backoff(
                    fetch_lora_model,
                    str(model_id),
                    lora_version,
                )
                if not main_page:
                    raise Exception(f"Failed to fetch model page for {model_id}")
                
                # Filter by base model
                base_model = lora_version.get('baseModel', 'Unknown')
                if base_model in ['SDXL 1.0', 'SD 2.0', 'Other', 'Unknown']:
                    logger.warning(f"Skipping {lora_display_name}: unsupported base model {base_model}")
                    await update_suggestion_status(version_id, "failed", f"Unsupported base model: {base_model}")
                    continue
                
                # Get preview image
                images = lora_version.get('images', [])
                if not images:
                    logger.warning(f"Skipping {lora_display_name}: no preview image")
                    await update_suggestion_status(version_id, "failed", "No preview image available")
                    continue
                image_url = select_preferred_preview_url(images)
                if not image_url:
                    logger.warning(f"Skipping {lora_display_name}: no static preview image")
                    await update_suggestion_status(version_id, "failed", "No static preview image available")
                    continue
                
                # Find safetensors file
                download_url = None
                hashes = None
                expected_hash = None
                
                for file_info in lora_version.get('files', []):
                    if (file_info['name'].lower().endswith('.safetensors') and 
                        file_info.get('pickleScanResult') == 'Success'):
                        download_url = file_info['downloadUrl']
                        hashes = file_info.get('hashes', {})
                        expected_hash = hashes.get('SHA256')
                        break
                
                if not download_url:
                    logger.warning(f"Skipping {lora_display_name}: no valid .safetensors file")
                    await update_suggestion_status(version_id, "failed", "No valid .safetensors file found")
                    continue
                
                # Build file path
                sanitized_version = sanitize_filename(lora_version['name'])
                sanitized_name = sanitize_filename(main_page['name'])
                storage_identity = sanitize_filename(str(lora_version['id']))
                filename = f"{sanitized_name}-{sanitized_version}-{storage_identity}.safetensors"
                
                base_model_folder = os.path.join(LORAS_FOLDER, sanitize_filename(base_model))
                lora_folder = os.path.join(base_model_folder, sanitized_name)
                file_path = os.path.join(lora_folder, filename)
                
                # Check if already exists
                if os.path.exists(file_path):
                    logger.info(f"File already exists: {filename}")
                else:
                    # Download with retry
                    logger.info(f"Downloading: {filename}")
                    success = await retry_with_backoff(
                        download_file_async, download_url, file_path, expected_hash
                    )
                    
                    if not success:
                        raise Exception("Download failed after retries")
                    
                    logger.info(f"Downloaded successfully: {filename}")
                
                # Build metadata
                metadata = {
                    "name": main_page['name'],
                    "version": lora_version['name'],
                    "base_model": base_model,
                    "download_url": download_url,
                    "is_nsfw": main_page.get('nsfw', False),
                    "is_minor": main_page.get('minor', False),
                    "creator": (main_page.get('creator') or {}).get('username'),
                    "description": main_page.get('description'),
                    "version_description": lora_version.get('description'),
                    "tags": main_page.get('tags', []),
                    "who_added": requestor,
                    "status": "downloaded",
                    "trigger_words": lora_version.get('trainedWords', []),
                    "hashes": hashes or {},
                    "image_url": image_url,
                    "file_path": file_path,
                    "version_id": lora_version['id']
                }
                
                # Upload metadata
                await upload_metadata(metadata)
                
                # Update suggestion status to downloaded
                await update_suggestion_status(version_id, "downloaded")

                # Notify the requesting user that their LoRA is live on-site.
                # Failures here must never prevent other downloads from proceeding.
                try:
                    await notify_lora_downloaded(
                        version_id=version_id,
                        name=main_page.get('name') or suggestion_name,
                        version=lora_version.get('name') or suggestion_version,
                        requestor=requestor,
                    )
                except Exception as notify_exc:
                    logger.warning(f"LoRA-ready push notification failed for {version_id}: {notify_exc}")

                logger.info(f"Completed: {lora_display_name}")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to process {lora_display_name}: {error_msg}")
                await update_suggestion_status(version_id, "failed", error_msg)
                # Continue with next LoRA instead of stopping
                continue
        
        logger.info("Download cycle completed")
        
    except Exception as e:
        logger.error(f"Critical error in download cycle: {e}")
    finally:
        is_downloading = False


# ============================================
# HTTP SERVER FOR REMOTE TRIGGER
# ============================================

async def handle_trigger(request):
    """Handle remote trigger request."""
    global trigger_event
    
    if is_downloading:
        return web.json_response({
            "status": "busy",
            "message": "Download already in progress"
        }, status=409)
    
    trigger_event.set()
    return web.json_response({
        "status": "triggered",
        "message": "Download cycle triggered"
    })


async def handle_status(request):
    """Handle status request - returns current downloading LoRA if any."""
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                # Check if anything is currently downloading
                await cur.execute("""
                    SELECT name, version, last_updated_date
                    FROM lora_suggestions
                    WHERE status = 'downloading'
                    LIMIT 1
                """)
                downloading = await cur.fetchone()
                
                # Count pending items
                await cur.execute("""
                    SELECT COUNT(*) FROM lora_suggestions WHERE status = 'approved'
                """)
                approved_count = (await cur.fetchone())[0]
                
                # Get last completed/failed
                await cur.execute("""
                    SELECT name, version, status, error_message, last_updated_date
                    FROM lora_suggestions
                    WHERE status IN ('downloaded', 'failed')
                    ORDER BY last_updated_date DESC
                    LIMIT 1
                """)
                last_processed = await cur.fetchone()
                
                if downloading:
                    return web.json_response({
                        "status": "downloading",
                        "current_lora": f"{downloading[0]} v{downloading[1]}",
                        "updated_at": downloading[2].isoformat() if downloading[2] else None,
                        "approved_count": approved_count,
                        "is_running": is_downloading
                    })
                else:
                    return web.json_response({
                        "status": "idle" if not is_downloading else "checking",
                        "current_lora": None,
                        "approved_count": approved_count,
                        "last_processed": {
                            "name": f"{last_processed[0]} v{last_processed[1]}" if last_processed else None,
                            "status": last_processed[2] if last_processed else None,
                            "error_message": last_processed[3] if last_processed else None,
                            "updated_at": last_processed[4].isoformat() if last_processed and last_processed[4] else None
                        } if last_processed else None,
                        "is_running": is_downloading
                    })
    except Exception as e:
        return web.json_response({
            "status": "error",
            "error": str(e)
        }, status=500)


async def handle_history(request):
    """Handle download history request - returns recently processed suggestions."""
    try:
        limit = int(request.query.get('limit', '20'))
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT name, version, status, error_message, version_id, last_updated_date
                    FROM lora_suggestions
                    WHERE status IN ('downloaded', 'failed', 'downloading')
                    ORDER BY last_updated_date DESC
                    LIMIT %s
                """, (limit,))
                rows = await cur.fetchall()
                
                history = [{
                    "lora_name": row[0],
                    "version": row[1],
                    "status": row[2],
                    "error_message": row[3],
                    "version_id": row[4],
                    "downloaded_at": row[5].isoformat() if row[5] else None
                } for row in rows]
                
                return web.json_response(history)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def start_http_server():
    """Start the HTTP server for remote triggering."""
    app = web.Application()
    app.router.add_post('/trigger', handle_trigger)
    app.router.add_get('/status', handle_status)
    app.router.add_get('/history', handle_history)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', DOWNLOADER_HTTP_PORT)
    await site.start()
    logger.info(f"HTTP server started on port {DOWNLOADER_HTTP_PORT}")
    return runner


# ============================================
# MAIN SERVICE LOOP
# ============================================

async def main():
    """Main service entry point."""
    global db_pool, http_session, trigger_event
    
    logger.info("=" * 60)
    logger.info("LoRA Downloader Service Starting")
    logger.info(f"LoRAs folder: {LORAS_FOLDER}")
    logger.info(f"Check interval: {DOWNLOAD_INTERVAL_SECONDS} seconds")
    logger.info(f"HTTP port: {DOWNLOADER_HTTP_PORT}")
    logger.info("=" * 60)
    
    # Validate configuration
    if not CIVITAI_API_KEY:
        logger.error("CIVITAI_API_KEY not set!")
        return
    
    if not all([DBHOST, DBNAME, DBUSER, DBPASS]):
        logger.error("Database configuration incomplete!")
        return
    
    # Create database pool
    async with psycopg_pool.AsyncConnectionPool(
        DSN,
        min_size=2,
        max_size=5,
        timeout=30,
        max_lifetime=3600,
        max_idle=300
    ) as pool:
        db_pool = pool
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for large files
        connector = aiohttp.TCPConnector(ssl=create_http_ssl_context())
        logger.info(f"Using CA bundle: {certifi.where()}")
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            http_session = session
            
            # Start HTTP server
            http_runner = await start_http_server()
            
            try:
                while True:
                    # Process downloads
                    await process_approved_loras()
                    
                    # Wait for interval or trigger
                    logger.info(f"Waiting {DOWNLOAD_INTERVAL_SECONDS}s for next check (or trigger)...")
                    trigger_event.clear()
                    
                    try:
                        await asyncio.wait_for(
                            trigger_event.wait(),
                            timeout=DOWNLOAD_INTERVAL_SECONDS
                        )
                        logger.info("Manual trigger received!")
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, proceed with scheduled check
                        
            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
            finally:
                await http_runner.cleanup()
                logger.info("Service stopped")


if __name__ == "__main__":
    asyncio.run(main())
