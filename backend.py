import os
import io
import base64
import sys
import asyncio
import hashlib
from typing import Optional, Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
import json
import re
import time
import math
import secrets
import tempfile
import uuid

# Fix for Windows - psycopg async requires SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
from pywebpush import webpush, WebPushException
import imagehash
import psycopg_pool
from psycopg import errors

from dynamic_prompts import (
    DynamicPromptError,
    expand_dynamic_prompt,
    public_library,
    preview_dynamic_prompt,
    wildcard_root_map,
)
from dynamic_prompts.library import (
    DEFAULT_CATEGORIES,
    DEFAULT_PREVIEW_COUNT,
    DEFAULT_STARTER_TEMPLATES,
    DEFAULT_WILDCARD_ITEMS,
    WILDCARD_SET_ID,
)
from helper_functions import *

PROFILING = False  # Set this from a settings model

logging.basicConfig(level=logging.ERROR)  # Configure logging

# Run 3 retries with exponential backoff strategy
load_dotenv()

API_KEY = os.environ.get("API_KEY")
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY")

DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")

# Define your connection parameters for PostgreSQL.
# - keepalives_*: detect dead peers (e.g. orphaned connections after a container
#   restart) within ~60s instead of waiting for the server-side 2h default.
# - options=-c ...: per-session safety timeouts so a stuck transaction cannot
#   hold row locks indefinitely and block the generator's pending-job polling.
DSN = (
    f"host={DBHOST} dbname='{DBNAME}' user={DBUSER} password={DBPASS} "
    "keepalives=1 keepalives_idle=30 keepalives_interval=10 keepalives_count=3 "
    "options='-c idle_in_transaction_session_timeout=30000 "
    "-c statement_timeout=60000 -c lock_timeout=5000'"
)

VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = os.environ.get("VAPID_CLAIMS")
# Windows Push Service rejects Web Push requests with TTL=0, so use a
# non-zero default unless explicitly overridden.
WEBPUSH_TTL_SECONDS = max(1, int(os.environ.get("WEBPUSH_TTL_SECONDS", "300")))
# Shared secret used by trusted local services (e.g., lora_downloader_service)
# to trigger server-initiated push notifications. Leave unset to disable.
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN")
# Destination opened when a user clicks a push notification.
PUBLIC_SITE_URL = (os.environ.get("PUBLIC_SITE_URL") or "https://mobians.ai/").rstrip("/") + "/"
LORAS_FOLDER = os.environ.get("LORAS_FOLDER", r"D:\mobians_api\loras")
ADMIN_LORA_MAX_UPLOAD_BYTES = max(
    1024 * 1024,
    int(os.environ.get("ADMIN_LORA_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))),
)
ADMIN_LORA_HEADER_MAX_BYTES = max(
    1024,
    int(os.environ.get("ADMIN_LORA_HEADER_MAX_BYTES", str(16 * 1024 * 1024))),
)
ADMIN_LORA_VERSION_ID_LOCK_KEY = 20498631

# Credit costs by model type
CREDIT_COSTS = {
    "SD 1.5": 10,      # sonicDiffusionV4
    "Pony": 15,        # autismMix (SDXL-based)
    "Illustrious": 15,  # novaFurryXL_ilV140 (SDXL-based), novaMobianXL_v10, novaMobianXL_v20
    "Anima": 20        # Anima-baseV1
}

# Additional cost per LoRA by model type
LORA_CREDIT_COSTS = {
    "SD 1.5": 2,
    "Pony": 5,
    "Illustrious": 5,
    "Anima": 5
}

# Upscale credit multiplier (upscales are computationally expensive)
UPSCALE_CREDIT_MULTIPLIER = 3

# Hi-res credit multiplier (generate + upscale in one job)
HIRES_CREDIT_MULTIPLIER = 4

# Job type for generate+upscale mode
HIRES_JOB_TYPE = "txt2img_upscale"

# Credit packages for purchase
CREDIT_PACKAGES = {
    "starter": {
        "id": "starter",
        "name": "Starter Pack",
        "price_usd": 5.00,
        "credits": 1500,
        "description": "1500 credits - Great for trying out priority queue"
    },
    "popular": {
        "id": "popular",
        "name": "Popular Pack",
        "price_usd": 10.00,
        "credits": 3500,
        "description": "3,500 credits - 17% bonus!"
    },
    "best_value": {
        "id": "best_value",
        "name": "Best Value Pack",
        "price_usd": 25.00,
        "credits": 10000,
        "description": "10,000 credits - 33% bonus!"
    }
}

# PayPal configuration
PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
PAYPAL_CLIENT_SECRET = os.environ.get("PAYPAL_CLIENT_SECRET")
PAYPAL_MODE = os.environ.get("PAYPAL_MODE", "sandbox")  # "sandbox" or "live"
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if PAYPAL_MODE == "sandbox" else "https://api-m.paypal.com"

# Map model names to their base types
MODEL_BASE_TYPES = {
    "sonicDiffusionV4": "SD 1.5",
    "autismMix": "Pony",
    "novaMobianXL_v10": "Illustrious",
    "novaFurryXL_ilV140": "Illustrious",
    "novaMobianXL_v20": "Illustrious",
    "Anima-baseV1": "Anima",
}

DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "novaMobianXL_v20")
LORA_SUGGESTION_LIMIT = 5
LORA_REREQUEST_COOLDOWN_DAYS = 7
SUPPORTED_LORA_BASE_MODELS = {"Pony", "SD 1.5", "Illustrious", "Anima"}


def is_supported_lora_base_model(base_model: Optional[str]) -> bool:
    return (base_model or "").strip() in SUPPORTED_LORA_BASE_MODELS


def normalize_model_id(model: Optional[str]) -> str:
    """Return a valid model id for generation.

    Clients can send stale/renamed model ids (e.g., from localStorage). To prevent
    jobs from getting stuck in the queue, coerce missing/unknown values to a safe
    default that exists in MODEL_BASE_TYPES.
    """
    available = list(MODEL_BASE_TYPES.keys())
    fallback = DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in MODEL_BASE_TYPES else (available[0] if available else "novaMobianXL_v20")

    if not model:
        return fallback

    if model in MODEL_BASE_TYPES:
        return model

    lower_map = {k.lower(): k for k in available}
    mapped = lower_map.get(model.lower())
    return mapped or fallback

app = FastAPI()
security = HTTPBearer(auto_error=False)
fastpass_cache = {}  # In-memory cache for FastPass data
civitai_link_cache: Dict[int, str] = {}
session = None
# Define db_pool as a global variable
db_pool = None

# Set up the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Create a connection pool
async def get_db_pool():
    return psycopg_pool.AsyncConnectionPool(DSN, min_size=2, max_size=10, timeout=10, max_lifetime=3600, max_idle=300)


@app.on_event("startup")
async def startup_event():
    global db_pool
    try:
        db_pool = await get_db_pool()
        print("Database pool initialized successfully")
    except Exception as e:
        print(f"Error initializing database pool: {e}")

    global session
    global r

    session = aiohttp.ClientSession(trust_env=True)

    # asyncio.create_task(refresh_fastpass_cache())
    
    # Start orphaned job cleanup background task
    asyncio.create_task(orphaned_job_cleanup_task())

    # Start the server-side push notifier so users get pinged when their
    # long-running upscale/hi-res jobs finish even if the tab is closed.
    asyncio.create_task(job_completion_notifier_task())


async def orphaned_job_cleanup_task():
    """Background task that periodically cleans up orphaned jobs and refunds credits."""
    # Wait a bit before first run to let the app fully start
    await asyncio.sleep(60)
    
    while True:
        try:
            async with db_pool.connection() as aconn:
                async with aconn.cursor() as acur:
                    # Call the cleanup function (cleans jobs older than 1 hour)
                    await acur.execute("SELECT * FROM cleanup_orphaned_jobs(1)")
                    result = await acur.fetchone()
                    if result and (result[0] > 0 or result[1] > 0):
                        logging.info(f"Orphaned job cleanup: {result[0]} jobs cleaned, {result[1]} credits refunded")
        except Exception as e:
            logging.error(f"Error in orphaned job cleanup task: {e}")
        
        # Run every 15 minutes
        await asyncio.sleep(900)


# How often the job-completion notifier polls for newly-finished jobs.
# Kept short so the UX feels near-real-time, but long enough that the query
# (which uses the `idx_generation_queue_pending_notify` partial index) stays
# cheap even under heavy traffic.
JOB_NOTIFY_POLL_SECONDS = 5

# Never look further back than this when claiming unnotified jobs. Protects
# the system from spamming stale notifications if the backend was down for a
# long time and a backlog of completed/failed rows built up.
JOB_NOTIFY_MAX_AGE_MINUTES = 30


async def _send_job_completion_notification(
    user_id: str,
    status: str,
    job_type: Optional[str],
    model: Optional[str],
    credit_cost: Optional[int],
    refunded: bool,
) -> None:
    """Build and dispatch the appropriate push payload for a finished job."""
    if status == "completed":
        if job_type == "upscale":
            title = "Your upscale is ready!"
            body = "Your high-resolution image is done — tap to view."
        elif job_type == HIRES_JOB_TYPE:
            title = "Your Hi-Res generation is ready!"
            body = "Tap to see your finished image."
        else:
            title = "Your image is ready!"
            body = "Tap to view your generation."
    elif status == "failed":
        title = "Your generation failed"
        if refunded and credit_cost:
            body = f"Something went wrong — {credit_cost} credits have been refunded."
        else:
            body = "Something went wrong. Please try again."
    else:
        return

    payload = {
        "notification": {
            "title": title,
            "body": body,
            "vibrate": [100, 50, 100],
            "data": {"url": PUBLIC_SITE_URL},
        }
    }
    await send_push_to_user(user_id, payload)


async def job_completion_notifier_task():
    """Poll for finished jobs owned by logged-in users and send a Web Push.

    Anonymous (not logged-in) jobs keep the existing client-triggered path in
    `NotificationService.sendPushNotification`. We only handle `user_id IS NOT NULL`
    here because those are the subscriptions we can actually target server-side.
    """
    # Small initial delay so the app fully starts and the pool is warm.
    await asyncio.sleep(15)

    while True:
        try:
            async with db_pool.connection() as aconn:
                async with aconn.cursor() as acur:
                    # Atomically claim the batch of jobs we're about to notify
                    # for, so two workers (or a restart) can't double-send.
                    await acur.execute(
                        """
                        UPDATE generation_queue
                        SET notified_at = NOW()
                        WHERE id IN (
                            SELECT id FROM generation_queue
                            WHERE notified_at IS NULL
                              AND status IN ('completed', 'failed')
                              AND user_id IS NOT NULL
                              AND create_date > NOW() - (%s || ' minutes')::interval
                            ORDER BY create_date ASC
                            LIMIT 50
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING user_id, status, job_type, model, credit_cost,
                                  COALESCE(refunded, FALSE)
                        """,
                        (JOB_NOTIFY_MAX_AGE_MINUTES,),
                    )
                    claimed = await acur.fetchall()
                    await aconn.commit()

            for row in claimed:
                user_id, status, job_type, model, credit_cost, refunded = row
                try:
                    await _send_job_completion_notification(
                        user_id=str(user_id),
                        status=status,
                        job_type=job_type,
                        model=model,
                        credit_cost=credit_cost,
                        refunded=refunded,
                    )
                except Exception as exc:
                    # One bad notification must not stop the poller.
                    logging.warning(f"Failed to send job notification: {exc}")

        except Exception as e:
            logging.error(f"Error in job_completion_notifier_task: {e}")

        await asyncio.sleep(JOB_NOTIFY_POLL_SECONDS)


@app.on_event("shutdown")
async def shutdown_event():
    global db_pool
    if db_pool:
        await db_pool.close()
    await session.close()
    # await app.state.db_pool.close()

@app.middleware("http")
async def add_cors_headers(request, call_next):
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = JSONResponse(content=None, status_code=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# Exception handler to ensure CORS headers on errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# Generic exception handler for unhandled errors
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    response = JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ============================================
# SESSION TOKEN AUTHENTICATION
# ============================================

def generate_session_token() -> str:
    """Generate a secure random session token."""
    return secrets.token_urlsafe(32)


async def create_session_token(user_id: str) -> str:
    """Create and store a session token for a user in the database.
    
    Allows multiple sessions per user (different devices/browsers).
    Old expired sessions are cleaned up periodically.
    """
    token = generate_session_token()
    expires_at = datetime.utcnow() + timedelta(days=30)
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Insert a new session token (allowing multiple per user)
            await acur.execute(
                """
                INSERT INTO user_sessions (user_id, token, expires_at)
                VALUES (%s, %s, %s)
                """,
                (user_id, token, expires_at)
            )
            
            # Clean up old expired sessions for this user (keep last 10 active ones)
            await acur.execute(
                """
                DELETE FROM user_sessions 
                WHERE user_id = %s AND (
                    expires_at < NOW() 
                    OR id NOT IN (
                        SELECT id FROM user_sessions 
                        WHERE user_id = %s AND expires_at > NOW()
                        ORDER BY created_at DESC 
                        LIMIT 10
                    )
                )
                """,
                (user_id, user_id)
            )
            await aconn.commit()
    return token


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    """
    Dependency to get the current user from session token.
    Returns None if no valid token is provided (allows anonymous access).
    """
    if not credentials:
        print("DEBUG: No credentials provided")
        return None
    
    token = credentials.credentials
    print(f"DEBUG: Token received: {token[:20]}..." if token and len(token) > 20 else f"DEBUG: Token: {token}")
    if not token:
        return None
    
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                # Look up the session token and get user data in one query
                await acur.execute(
                    """
                    SELECT u.id, u.discord_user_id, u.google_user_id, u.email, u.username, 
                           u.display_name, u.avatar_url, u.credits, u.last_daily_bonus, 
                           u.daily_bonus_streak, u.is_banned
                    FROM users u
                    JOIN user_sessions s ON u.id = s.user_id
                    WHERE s.token = %s AND s.expires_at > NOW()
                    """,
                    (token,)
                )
                row = await acur.fetchone()
                print(f"DEBUG: Query result row: {row}")
                if not row:
                    return None
                
                return {
                    "user_id": str(row[0]),
                    "discord_user_id": row[1],
                    "google_user_id": row[2],
                    "email": row[3],
                    "username": row[4],
                    "display_name": row[5],
                    "avatar_url": row[6],
                    "credits": row[7],
                    "last_daily_bonus": row[8].isoformat() if row[8] else None,
                    "daily_bonus_streak": row[9],
                    "is_banned": row[10]
                }
    except Exception as e:
        logging.error(f"Error fetching user: {e}")
        return None


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Dependency that requires authentication - raises 401 if not authenticated.
    """
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if user.get("is_banned"):
        raise HTTPException(status_code=403, detail="Account is banned")
    return user


def get_lora_requestor_candidates(user: Optional[dict]) -> List[str]:
    candidates: List[str] = []
    if not user:
        return candidates

    for key in ("user_id", "discord_user_id", "google_user_id", "username", "email"):
        value = user.get(key)
        if value is None:
            continue

        normalized = str(value).strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    return candidates


def get_primary_lora_requestor(user: Optional[dict]) -> str:
    candidates = get_lora_requestor_candidates(user)
    if not candidates:
        raise HTTPException(status_code=400, detail="No requestor id available")
    return candidates[0]


async def upsert_user(
    discord_user_id: str = None,
    google_user_id: str = None,
    email: str = None,
    username: str = None,
    display_name: str = None,
    avatar_url: str = None
) -> dict:
    """
    Create or update a user. Returns user data with token.
    - If user exists (by discord_user_id or google_user_id), update their info
    - If new user, create with 100 bonus credits
    - Handles account linking (adding google to existing discord account, etc.)
    """
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # First, try to find existing user by either OAuth ID
            existing_user = None
            
            if discord_user_id:
                await acur.execute(
                    "SELECT id, credits, last_daily_bonus, daily_bonus_streak FROM users WHERE discord_user_id = %s",
                    (discord_user_id,)
                )
                existing_user = await acur.fetchone()
            
            if not existing_user and google_user_id:
                await acur.execute(
                    "SELECT id, credits, last_daily_bonus, daily_bonus_streak FROM users WHERE google_user_id = %s",
                    (google_user_id,)
                )
                existing_user = await acur.fetchone()
            
            if existing_user:
                # Update existing user
                user_id = existing_user[0]
                credits = existing_user[1]
                last_daily_bonus = existing_user[2]
                daily_bonus_streak = existing_user[3]
                
                # Build update query dynamically to handle linking
                updates = ["last_login_at = NOW()"]
                params = []
                
                if discord_user_id:
                    updates.append("discord_user_id = COALESCE(discord_user_id, %s)")
                    params.append(discord_user_id)
                if google_user_id:
                    updates.append("google_user_id = COALESCE(google_user_id, %s)")
                    params.append(google_user_id)
                if email:
                    updates.append("email = COALESCE(%s, email)")
                    params.append(email)
                if display_name:
                    updates.append("display_name = %s")
                    params.append(display_name)
                if avatar_url:
                    updates.append("avatar_url = %s")
                    params.append(avatar_url)
                
                params.append(user_id)
                
                await acur.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE id = %s",
                    tuple(params)
                )
                
                is_new_user = False
            else:
                # Create new user with signup bonus
                await acur.execute(
                    """
                    INSERT INTO users (discord_user_id, google_user_id, email, username, display_name, avatar_url, credits, last_login_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 100, NOW())
                    RETURNING id, credits
                    """,
                    (discord_user_id, google_user_id, email, username, display_name, avatar_url)
                )
                result = await acur.fetchone()
                user_id = result[0]
                credits = result[1]
                last_daily_bonus = None
                daily_bonus_streak = 0
                
                # Record signup bonus transaction
                await acur.execute(
                    """
                    INSERT INTO credit_transactions (user_id, amount, balance_after, transaction_type, description)
                    VALUES (%s, 100, 100, 'signup_bonus', 'Welcome bonus - 100 free credits!')
                    """,
                    (user_id,)
                )
                
                is_new_user = True
            
            # Commit the user creation/update before creating session token
            # This is required because create_session_token uses a separate connection
            await aconn.commit()
            
            # Generate session token and store in database
            token = await create_session_token(str(user_id))
            
            return {
                "user_id": str(user_id),
                "discord_user_id": discord_user_id,
                "google_user_id": google_user_id,
                "email": email,
                "username": username,
                "display_name": display_name,
                "avatar_url": avatar_url,
                "credits": credits,
                "last_daily_bonus": last_daily_bonus.isoformat() if last_daily_bonus else None,
                "daily_bonus_streak": daily_bonus_streak,
                "token": token,
                "is_new_user": is_new_user
            }


def get_credit_cost(model: str, loras: Optional[List[Dict[str, Any]]] = None) -> int:
    """Get the credit cost for a given model and optional LoRAs."""
    normalized_model = normalize_model_id(model)
    base_type = MODEL_BASE_TYPES.get(normalized_model, "SD 1.5")
    base_cost = CREDIT_COSTS.get(base_type, CREDIT_COSTS.get("SD 1.5", 0))
    lora_count = len(loras) if isinstance(loras, list) else 0
    per_lora_cost = LORA_CREDIT_COSTS.get(base_type, 0)
    return base_cost + (lora_count * per_lora_cost)

def get_upscale_credit_cost(model: str, loras: Optional[List[Dict[str, Any]]] = None) -> int:
    """Get the credit cost for an upscale job (base model cost * multiplier + LoRA costs * 3)."""
    normalized_model = normalize_model_id(model)
    base_type = MODEL_BASE_TYPES.get(normalized_model, "SD 1.5")
    base_cost = CREDIT_COSTS.get(base_type, CREDIT_COSTS.get("SD 1.5", 0))
    lora_count = len(loras) if isinstance(loras, list) else 0
    per_lora_cost = LORA_CREDIT_COSTS.get(base_type, 0)
    lora_total = lora_count * per_lora_cost * UPSCALE_CREDIT_MULTIPLIER
    return (base_cost * UPSCALE_CREDIT_MULTIPLIER) + lora_total


def get_hires_credit_cost(model: str, loras: Optional[List[Dict[str, Any]]] = None) -> int:
    """Get the credit cost for a hi-res (generate+upscale) job (base model cost * 4 + LoRA costs * 4)."""
    normalized_model = normalize_model_id(model)
    base_type = MODEL_BASE_TYPES.get(normalized_model, "SD 1.5")
    base_cost = CREDIT_COSTS.get(base_type, CREDIT_COSTS.get("SD 1.5", 0))
    lora_count = len(loras) if isinstance(loras, list) else 0
    per_lora_cost = LORA_CREDIT_COSTS.get(base_type, 0)
    lora_total = lora_count * per_lora_cost * HIRES_CREDIT_MULTIPLIER
    return (base_cost * HIRES_CREDIT_MULTIPLIER) + lora_total


class ImageData(BaseModel):
    url: Optional[str] = None
    width: Optional[int]
    height: Optional[int]
    aspectRatio: Optional[str]
    base64: str
    UUID: Optional[str]
    rated: Optional[bool]


class DynamicPromptingConfig(BaseModel):
    enabled: bool = False
    mode: Optional[str] = "random"
    template: Optional[str] = None
    wildcard_set: Optional[str] = "mobians-v1"
    preview_count: Optional[int] = 4
    max_generations: Optional[int] = 32
    expansion_seed: Optional[int] = None
    selected_preview_index: Optional[int] = 0


class DynamicPromptPreviewRequest(BaseModel):
    template: str
    mode: Optional[str] = "random"
    seed: Optional[int] = None
    preview_count: Optional[int] = 4
    max_generations: Optional[int] = 32


class DynamicPromptAdminCategoryUpdate(BaseModel):
    id: str
    label: str
    description: Optional[str] = ""
    token: Optional[str] = None
    display_order: Optional[int] = 0
    is_active: Optional[bool] = True
    entries: Optional[List[str]] = None


class DynamicPromptAdminStarterTemplateUpdate(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    token: Optional[str] = None
    template: str
    display_order: Optional[int] = 0
    is_active: Optional[bool] = True


class DynamicPromptAdminLibraryUpdate(BaseModel):
    categories: List[DynamicPromptAdminCategoryUpdate]
    starter_templates: List[DynamicPromptAdminStarterTemplateUpdate] = []


class DynamicPromptTemplateCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    template: str
    tags: Optional[List[str]] = []


class DynamicPromptTemplateUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    template: Optional[str] = None
    tags: Optional[List[str]] = None


class DynamicPromptTemplateRejectRequest(BaseModel):
    reason: str


class DynamicPromptCustomCategoryCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    entries: Optional[List[str]] = []
    tags: Optional[List[str]] = []


class DynamicPromptCustomCategoryUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    entries: Optional[List[str]] = None
    tags: Optional[List[str]] = None


COMMUNITY_TEMPLATE_STATUSES = {"private", "pending", "approved", "rejected", "hidden"}
ADMIN_COMMUNITY_TEMPLATE_STATUSES = {"pending", "approved", "rejected", "hidden"}
COMMUNITY_TEMPLATE_MUTABLE_STATUSES = {"private", "pending", "approved", "rejected"}
COMMUNITY_TEMPLATE_PUBLIC_SORTS = {"new", "top", "popular"}
COMMUNITY_TEMPLATE_TITLE_MAX = 90
COMMUNITY_TEMPLATE_DESCRIPTION_MAX = 500
COMMUNITY_TEMPLATE_MAX_TAGS = 8
COMMUNITY_TEMPLATE_TAG_MAX = 24
COMMUNITY_TEMPLATE_PREVIEW_SEED = 1729
CUSTOM_CATEGORY_STATUSES = {"private", "public", "hidden"}
ADMIN_CUSTOM_CATEGORY_STATUSES = {"public", "hidden"}
CUSTOM_CATEGORY_PUBLIC_SORTS = {"new", "top", "popular"}
CUSTOM_CATEGORY_TITLE_MAX = 80
CUSTOM_CATEGORY_DESCRIPTION_MAX = 500
CUSTOM_CATEGORY_MAX_ENTRIES = 250
CUSTOM_CATEGORY_ENTRY_MAX = 180
DYNAMIC_PROMPT_VOTE_CONTENT_TYPES = {"template", "category"}
DYNAMIC_PROMPT_CREATOR_VOTE_REWARD_CREDITS = 50
DYNAMIC_PROMPT_VOTER_VOTE_REWARD_CREDITS = 15
DYNAMIC_PROMPT_TEMPLATE_VOTER_DAILY_CAP = 3
DYNAMIC_PROMPT_CATEGORY_VOTER_DAILY_CAP = 3
DYNAMIC_PROMPT_TEMPLATE_CREATOR_TRANSACTION_TYPE = "dynamic_template_vote_received"
DYNAMIC_PROMPT_TEMPLATE_VOTER_TRANSACTION_TYPE = "dynamic_template_vote_given"
DYNAMIC_PROMPT_CATEGORY_CREATOR_TRANSACTION_TYPE = "dynamic_category_vote_received"
DYNAMIC_PROMPT_CATEGORY_VOTER_TRANSACTION_TYPE = "dynamic_category_vote_given"


class JobData(BaseModel):
    prompt: str
    image: Optional[str] = None
    image_UUID: Optional[str] = None
    mask_image: Optional[str] = None
    color_inpaint: Optional[bool] = None
    control_image: Optional[str] = None
    scheduler: int
    steps: int
    negative_prompt: str
    width: int
    height: int
    guidance_scale: int
    seed: int
    batch_size: int
    strength: Optional[float] = None
    job_type: str
    model: Optional[str] = None
    fast_pass_code: Optional[str] = None
    rating: Optional[bool] = None
    enable_upscale: Optional[bool] = False
    is_dev_job: Optional[bool] = False
    loras: Optional[List[Dict[str, Any]]] = None
    lossy_images: Optional[bool] = False
    regional_prompting: Optional[Dict[str, Any]] = None
    dynamic_prompting: Optional[DynamicPromptingConfig] = None
    # New fields for credit system
    queue_type: Optional[str] = "free"  # "free" or "priority"


class ImageRequestModel(JobData):
    image: Optional[str] = None
    fast_pass_enabled: Optional[bool] = False
    user_id: Optional[str] = None
    credit_cost: Optional[int] = 0


DYNAMIC_PROMPT_CACHE_TTL_SECONDS = 300
_dynamic_prompt_cache: Dict[str, Any] = {
    "expires_at": 0.0,
    "library": None,
    "root_map": None,
}


def _invalidate_dynamic_prompt_cache() -> None:
    _dynamic_prompt_cache["expires_at"] = 0.0
    _dynamic_prompt_cache["library"] = None
    _dynamic_prompt_cache["root_map"] = None


def _fallback_dynamic_prompt_assets() -> Dict[str, Any]:
    fallback_categories = []
    for category_order, category in enumerate(DEFAULT_CATEGORIES):
        item_values = DEFAULT_WILDCARD_ITEMS.get(category["id"], [])
        fallback_categories.append(
            {
                **category,
                "examples": item_values[:3],
                "items": [
                    {
                        "value": value,
                        "display_order": item_order,
                        "is_active": True,
                    }
                    for item_order, value in enumerate(item_values)
                ],
                "display_order": category_order,
                "is_active": True,
            }
        )

    fallback_starters = [
        {
            **starter,
            "display_order": starter_order,
            "is_active": True,
        }
        for starter_order, starter in enumerate(DEFAULT_STARTER_TEMPLATES)
    ]
    return {
        "library": public_library(),
        "root_map": wildcard_root_map(),
        "admin_library": {
            "wildcard_set": WILDCARD_SET_ID,
            "categories": fallback_categories,
            "starter_templates": fallback_starters,
        },
    }


async def _load_dynamic_prompt_assets(
    include_inactive: bool = False,
    user_id: Optional[str] = None,
    include_public_custom: bool = True,
) -> Dict[str, Any]:
    can_use_cache = not include_inactive and not user_id and not include_public_custom
    if can_use_cache and _dynamic_prompt_cache["expires_at"] > time.time():
        return {
            "library": _dynamic_prompt_cache["library"],
            "root_map": _dynamic_prompt_cache["root_map"],
        }

    custom_category_rows = []
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                category_active_clause = "" if include_inactive else "AND is_active = TRUE"
                item_active_clause = "" if include_inactive else "AND i.is_active = TRUE AND c.is_active = TRUE"
                starter_active_clause = "" if include_inactive else "AND is_active = TRUE"

                await acur.execute(
                    f"""
                    SELECT id, label, token, description, display_order, is_active
                    FROM dynamic_prompt.categories
                    WHERE wildcard_set = %s {category_active_clause}
                    ORDER BY display_order, label
                    """,
                    (WILDCARD_SET_ID,),
                )
                category_rows = await acur.fetchall()

                await acur.execute(
                    f"""
                    SELECT i.category_id, i.value, i.display_order, i.is_active
                    FROM dynamic_prompt.items i
                    JOIN dynamic_prompt.categories c ON c.id = i.category_id
                    WHERE c.wildcard_set = %s {item_active_clause}
                    ORDER BY i.category_id, i.display_order, i.value
                    """,
                    (WILDCARD_SET_ID,),
                )
                item_rows = await acur.fetchall()

                await acur.execute(
                    f"""
                    SELECT id, name, description, token, template, display_order, is_active
                    FROM dynamic_prompt.starter_templates
                    WHERE wildcard_set = %s {starter_active_clause}
                    ORDER BY display_order, name
                    """,
                    (WILDCARD_SET_ID,),
                )
                starter_rows = await acur.fetchall()

                if not include_inactive and (user_id or include_public_custom):
                    custom_where_clauses = ["c.status = 'public'"] if include_public_custom else []
                    custom_params: List[Any] = []
                    if user_id:
                        custom_where_clauses.append("(c.user_id = %s AND c.status <> 'hidden')")
                        custom_params.append(user_id)
                    await acur.execute(
                        f"""
                        SELECT
                            c.id,
                            c.title,
                            c.description,
                            c.token,
                            COALESCE((
                                SELECT ARRAY_AGG(ci.value ORDER BY ci.display_order, ci.value)
                                FROM dynamic_prompt.custom_category_items ci
                                WHERE ci.category_id = c.id AND ci.is_active = TRUE
                            ), ARRAY[]::text[]) AS entries
                        FROM dynamic_prompt.custom_categories c
                        WHERE {' OR '.join(custom_where_clauses)}
                        ORDER BY c.updated_at DESC, c.title
                        LIMIT 150
                        """,
                        tuple(custom_params),
                    )
                    custom_category_rows = await acur.fetchall()
    except (errors.InvalidSchemaName, errors.UndefinedTable, errors.UndefinedColumn) as exc:
        logging.warning("Dynamic prompt DB tables are unavailable; using built-in seed data: %s", exc)
        return _fallback_dynamic_prompt_assets()

    items_by_category: Dict[str, List[str]] = {}
    admin_items_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for category_id, value, display_order, is_active in item_rows:
        items_by_category.setdefault(category_id, [])
        if is_active:
            items_by_category[category_id].append(value)
        admin_items_by_category.setdefault(category_id, []).append(
            {
                "value": value,
                "display_order": display_order,
                "is_active": bool(is_active),
            }
        )

    categories = []
    admin_categories = []
    for category_id, label, token, description, display_order, is_active in category_rows:
        active_items = items_by_category.get(category_id, [])
        category = {
            "id": category_id,
            "label": label,
            "token": token,
            "description": description,
            "examples": active_items[:3],
        }
        categories.append(category)
        admin_categories.append(
            {
                **category,
                "items": admin_items_by_category.get(category_id, []),
                "display_order": display_order,
                "is_active": bool(is_active),
            }
        )

    for category_id, title, description, token, entries in custom_category_rows:
        entry_values = [str(entry).strip() for entry in (entries or []) if str(entry).strip()]
        if not entry_values:
            continue
        wildcard_id = _wildcard_id_from_token(token)
        items_by_category[wildcard_id] = entry_values
        categories.append(
            {
                "id": wildcard_id,
                "label": title,
                "token": token,
                "description": description or "Custom prompt category",
                "examples": entry_values[:3],
                "source": "custom",
                "category_id": str(category_id),
            }
        )

    starter_templates = [
        {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "token": row[3] or _starter_template_token(row[0]),
            "template": row[4],
            "display_order": row[5],
            "is_active": bool(row[6]),
        }
        for row in starter_rows
    ]
    active_starters = [
        {
            "id": starter["id"],
            "name": starter["name"],
            "description": starter["description"],
            "token": starter["token"],
            "template": starter["template"],
        }
        for starter in starter_templates
        if starter["is_active"]
    ]

    library = public_library(categories, active_starters)
    root_map = wildcard_root_map(items_by_category)
    assets = {
        "library": library,
        "root_map": root_map,
        "admin_library": {
            "wildcard_set": WILDCARD_SET_ID,
            "categories": admin_categories,
            "starter_templates": starter_templates,
        },
    }

    if can_use_cache:
        _dynamic_prompt_cache["library"] = library
        _dynamic_prompt_cache["root_map"] = root_map
        _dynamic_prompt_cache["expires_at"] = time.time() + DYNAMIC_PROMPT_CACHE_TTL_SECONDS

    return assets


def _dynamic_config_dict(config: Optional[DynamicPromptingConfig]) -> Dict[str, Any]:
    if not config:
        return {}
    return {key: value for key, value in config.dict().items() if value is not None}


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _has_dynamic_prompt_variant_syntax(template: str) -> bool:
    return bool(re.search(r"\{[^{}]*\|[^{}]*\}", template))


def _has_dynamic_prompt_syntax(template: str, allowed_wildcards: Optional[set[str]] = None) -> bool:
    if _has_dynamic_prompt_variant_syntax(template):
        return True
    category_ids = _extract_dynamic_prompt_wildcard_ids(template)
    if category_ids:
        if allowed_wildcards is None:
            return True
        if any(category_id in allowed_wildcards for category_id in category_ids):
            return True
    return bool(_extract_dynamic_prompt_template_ids(template))


def _extract_dynamic_prompt_wildcard_ids(template: str) -> List[str]:
    return sorted(set(re.findall(r"(?<!_)_([A-Za-z0-9](?:[-\w/]*[A-Za-z0-9])?)_(?!_)", template)))


def _extract_dynamic_prompt_template_ids(template: str) -> List[str]:
    return sorted(set(re.findall(r"__([-\w/]+)__", template)))


def _category_tokens_to_engine_wildcards(template: str) -> str:
    return re.sub(r"(?<!_)_([A-Za-z0-9](?:[-\w/]*[A-Za-z0-9])?)_(?!_)", r"__\1__", template)


def _allowed_dynamic_prompt_wildcard_ids(assets: Dict[str, Any]) -> set[str]:
    wildcard_ids: set[str] = set()
    for category in assets.get("library", {}).get("categories", []):
        category_id = str(category.get("id", "")).strip()
        if category_id:
            wildcard_ids.add(category_id)
    return wildcard_ids


def _resolve_dynamic_prompt_request(
    prompt: str,
    config: Optional[DynamicPromptingConfig],
    allowed_wildcards: Optional[set[str]] = None,
) -> Optional[Tuple[str, DynamicPromptingConfig]]:
    prompt_text = str(prompt or "").strip()
    resolved_config = config or DynamicPromptingConfig()
    config_template = str(resolved_config.template or "").strip()

    candidates: List[str] = []
    if resolved_config.enabled and config_template:
        candidates.append(config_template)
    if prompt_text:
        candidates.append(prompt_text)

    for candidate in candidates:
        if (resolved_config.enabled and candidate == config_template) or _has_dynamic_prompt_syntax(candidate, allowed_wildcards):
            resolved_config.enabled = True
            resolved_config.template = candidate
            return candidate, resolved_config

    return None


async def _resolve_dynamic_prompt_template_tokens(
    template: str,
    user_id: Optional[str],
    assets: Dict[str, Any],
) -> str:
    template_ids = _extract_dynamic_prompt_template_ids(template)
    if not template_ids:
        return template

    templates_by_id: Dict[str, str] = {}
    for starter in assets.get("library", {}).get("starter_templates", []):
        starter_token = str(starter.get("token") or _starter_template_token(str(starter.get("id") or ""))).strip()
        starter_id = _wildcard_id_from_token(starter_token)
        if starter_id in template_ids:
            templates_by_id[starter_id] = str(starter.get("template") or "").strip()

    missing_ids = [template_id for template_id in template_ids if template_id not in templates_by_id]
    if missing_ids and db_pool is not None:
        try:
            async with db_pool.connection() as aconn:
                async with aconn.cursor() as acur:
                    await acur.execute(
                        """
                        SELECT token, template
                        FROM dynamic_prompt.templates
                        WHERE token = ANY(%s)
                          AND status <> 'hidden'
                          AND (status = 'approved' OR user_id = %s)
                        """,
                        ([f"__{template_id}__" for template_id in missing_ids], user_id),
                    )
                    for token, template_body in await acur.fetchall():
                        templates_by_id[_wildcard_id_from_token(token)] = str(template_body or "").strip()
        except (errors.InvalidSchemaName, errors.UndefinedTable, errors.UndefinedColumn) as exc:
            logging.warning("Dynamic prompt template tokens are unavailable: %s", exc)

    unresolved_ids = [template_id for template_id in template_ids if template_id not in templates_by_id]
    if unresolved_ids:
        raise HTTPException(status_code=400, detail=f"Unknown template token: __{unresolved_ids[0]}__")

    for template_id, template_body in templates_by_id.items():
        nested_ids = _extract_dynamic_prompt_template_ids(template_body)
        if nested_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Template token __{template_id}__ cannot contain another template token.",
            )

    def replace_template_token(match: re.Match[str]) -> str:
        template_id = match.group(1)
        return templates_by_id.get(template_id, match.group(0))

    return re.sub(r"__([-\w/]+)__", replace_template_token, template)


async def _prepare_dynamic_prompt_template_for_expansion(
    template: str,
    user_id: Optional[str],
    assets: Dict[str, Any],
    allow_template_tokens: bool = True,
) -> str:
    clean_template = str(template or "").strip()
    template_ids = _extract_dynamic_prompt_template_ids(clean_template)
    if template_ids and not allow_template_tokens:
        raise HTTPException(
            status_code=400,
            detail="Templates can include categories and variants, but not other template tokens.",
        )

    resolved_template = clean_template
    if template_ids:
        resolved_template = await _resolve_dynamic_prompt_template_tokens(clean_template, user_id, assets)

    allowed_wildcards = _allowed_dynamic_prompt_wildcard_ids(assets)
    unknown_wildcards = [
        wildcard
        for wildcard in _extract_dynamic_prompt_wildcard_ids(resolved_template)
        if wildcard not in allowed_wildcards
    ]
    if unknown_wildcards:
        raise HTTPException(status_code=400, detail=f"Unknown category token: _{unknown_wildcards[0]}_")

    return _category_tokens_to_engine_wildcards(resolved_template)


def _normalize_community_template_tags(tags: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for raw_tag in tags or []:
        tag = re.sub(r"\s+", "-", str(raw_tag).strip().lower())
        tag = re.sub(r"[^a-z0-9_-]", "", tag)
        if not tag:
            continue
        if len(tag) > COMMUNITY_TEMPLATE_TAG_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"Tags must be {COMMUNITY_TEMPLATE_TAG_MAX} characters or fewer.",
            )
        if tag not in normalized:
            normalized.append(tag)
        if len(normalized) > COMMUNITY_TEMPLATE_MAX_TAGS:
            raise HTTPException(
                status_code=400,
                detail=f"Use {COMMUNITY_TEMPLATE_MAX_TAGS} tags or fewer.",
            )
    return normalized


CUSTOM_CATEGORY_SLUG_MAX = 56
CUSTOM_TEMPLATE_SLUG_MAX = 56
CUSTOM_CATEGORY_NAMESPACE_MAX = 32
RESERVED_DYNAMIC_PROMPT_NAMESPACES = {
    "admin",
    "api",
    "community",
    "custom",
    "dynamic",
    "dynamic-prompt",
    "dynamic-prompts",
    "mobian",
    "public",
    "system",
}


def _dynamic_prompt_slug_part(value: Any, max_length: int, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        slug = fallback
    return slug[:max_length].strip("-") or fallback


def _custom_category_slug(title: str) -> str:
    return _dynamic_prompt_slug_part(title, CUSTOM_CATEGORY_SLUG_MAX, "category")


def _custom_template_slug(title: str) -> str:
    return _dynamic_prompt_slug_part(title, CUSTOM_TEMPLATE_SLUG_MAX, "template")


def _custom_category_id_suffix(category_id: str, length: int = 12) -> str:
    suffix = re.sub(r"[^a-f0-9]", "", str(category_id or "").lower())
    return suffix[:length] or "custom"


def _namespace_letters_and_numbers(value: Any) -> str:
    namespace = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    return namespace[:CUSTOM_CATEGORY_NAMESPACE_MAX]


def _display_name_namespace_slug(display_name: Any) -> str:
    display_value = str(display_name or "").strip()
    if "@" in display_value:
        return ""
    return _namespace_letters_and_numbers(display_value)


def _username_namespace_slug(username: Any) -> str:
    username_value = str(username or "").strip()
    if "@" in username_value:
        username_value = username_value.split("@", 1)[0]
    return _namespace_letters_and_numbers(username_value)


def _custom_category_namespace_candidates(user: Optional[Dict[str, Any]]) -> List[str]:
    user = user or {}
    user_id = str(user.get("user_id") or "")
    fallback_short = f"user-{_custom_category_id_suffix(user_id, 8)}"
    fallback_long = f"user-{_custom_category_id_suffix(user_id, 12)}"
    candidates = [
        _display_name_namespace_slug(user.get("display_name")),
        _username_namespace_slug(user.get("username")),
        fallback_short,
        fallback_long,
    ]

    namespaces: List[str] = []
    for candidate in candidates:
        if not candidate or candidate in RESERVED_DYNAMIC_PROMPT_NAMESPACES or candidate in namespaces:
            continue
        namespaces.append(candidate)
    return namespaces or [fallback_short]


def _custom_category_namespace_slug(user: Optional[Dict[str, Any]]) -> str:
    return _custom_category_namespace_candidates(user)[0]


def _custom_category_token_from_parts(namespace: str, slug: str) -> str:
    namespace_slug = _dynamic_prompt_slug_part(namespace, CUSTOM_CATEGORY_NAMESPACE_MAX, "user")
    category_slug = _custom_category_slug(slug)
    return f"_{namespace_slug}/{category_slug}_"


def _template_token_from_parts(namespace: str, slug: str) -> str:
    namespace_slug = _dynamic_prompt_slug_part(namespace, CUSTOM_CATEGORY_NAMESPACE_MAX, "user")
    template_slug = _custom_template_slug(slug)
    return f"__{namespace_slug}/{template_slug}__"


def _starter_template_token(starter_id: str) -> str:
    return _template_token_from_parts("mobian", starter_id)


def _legacy_custom_category_token(category_id: str) -> str:
    return f"_custom/{category_id}_"


def _custom_category_token(category_id: str, title: Optional[str] = None, namespace: str = "custom") -> str:
    if title is None:
        return _legacy_custom_category_token(category_id)
    return _custom_category_token_from_parts(namespace, _custom_category_slug(title))


def _wildcard_id_from_token(token: Any) -> str:
    return str(token or "").strip().strip("_")


def _namespace_from_category_token(token: Any) -> str:
    wildcard_id = _wildcard_id_from_token(token)
    namespace, separator, _category_slug = wildcard_id.partition("/")
    return namespace if separator else ""


async def _custom_category_namespace_for_user(acur: Any, user: Dict[str, Any]) -> str:
    user_id = str(user.get("user_id") or "")
    await acur.execute(
        """
        SELECT token
        FROM dynamic_prompt.custom_categories
        WHERE user_id = %s AND token !~ '^_custom/'
        ORDER BY created_at ASC, id ASC
        LIMIT 1
        """,
        (user_id,),
    )
    existing_namespace_row = await acur.fetchone()
    if existing_namespace_row:
        existing_namespace = _namespace_from_category_token(existing_namespace_row[0])
        if existing_namespace:
            return existing_namespace

    for namespace in _custom_category_namespace_candidates(user):
        await acur.execute(
            """
            SELECT 1
            FROM dynamic_prompt.custom_categories
            WHERE user_id <> %s AND token LIKE %s ESCAPE '\\'
            LIMIT 1
            """,
            (user_id, f"\\_{namespace}/%"),
        )
        if not await acur.fetchone():
            return namespace

    return f"user-{_custom_category_id_suffix(user_id, 16)}"


async def _generate_unique_custom_category_token(
    acur: Any,
    title: str,
    category_id: str,
    user: Dict[str, Any],
    exclude_category_id: Optional[str] = None,
) -> str:
    namespace = await _custom_category_namespace_for_user(acur, user)
    base_slug = _custom_category_slug(title)
    id_suffix = _custom_category_id_suffix(category_id)
    candidate_slugs = [base_slug, f"{base_slug}-{id_suffix}"]
    candidate_slugs.extend(f"{base_slug}-{id_suffix}-{index}" for index in range(2, 100))

    for slug in candidate_slugs:
        candidate = _custom_category_token_from_parts(namespace, slug)
        if exclude_category_id:
            await acur.execute(
                """
                SELECT 1
                FROM dynamic_prompt.custom_categories
                WHERE id <> %s
                  AND token = %s
                LIMIT 1
                """,
                (exclude_category_id, candidate),
            )
        else:
            await acur.execute(
                """
                SELECT 1
                FROM dynamic_prompt.custom_categories
                WHERE token = %s
                LIMIT 1
                """,
                (candidate,),
            )
        if not await acur.fetchone():
            return candidate

    raise HTTPException(status_code=500, detail="Unable to create a unique category token.")


async def _generate_unique_template_token(
    acur: Any,
    title: str,
    template_id: str,
    user: Dict[str, Any],
    exclude_template_id: Optional[str] = None,
) -> str:
    namespace = _custom_category_namespace_slug(user)
    base_slug = _custom_template_slug(title)
    id_suffix = _custom_category_id_suffix(template_id)
    candidate_slugs = [base_slug, f"{base_slug}-{id_suffix}"]
    candidate_slugs.extend(f"{base_slug}-{id_suffix}-{index}" for index in range(2, 100))

    for slug in candidate_slugs:
        candidate = _template_token_from_parts(namespace, slug)
        await acur.execute(
            """
            SELECT 1
            FROM dynamic_prompt.starter_templates
            WHERE token = %s
            LIMIT 1
            """,
            (candidate,),
        )
        if await acur.fetchone():
            continue

        if exclude_template_id:
            await acur.execute(
                """
                SELECT 1
                FROM dynamic_prompt.templates
                WHERE id <> %s
                  AND token = %s
                LIMIT 1
                """,
                (exclude_template_id, candidate),
            )
        else:
            await acur.execute(
                """
                SELECT 1
                FROM dynamic_prompt.templates
                WHERE token = %s
                LIMIT 1
                """,
                (candidate,),
            )
        if not await acur.fetchone():
            return candidate

    raise HTTPException(status_code=500, detail="Unable to create a unique template token.")


async def _resolve_updated_custom_category_token(
    acur: Any,
    user: Dict[str, Any],
    category_id: str,
    title: str,
    current_token: Optional[str],
 ) -> str:
    clean_current_token = str(current_token or "").strip()
    clean_title = str(title or "").strip()

    if not clean_current_token or not clean_title:
        return clean_current_token

    return await _generate_unique_custom_category_token(
        acur,
        clean_title,
        category_id,
        user,
        exclude_category_id=category_id,
    )


def _normalize_custom_category_entries(entries: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for raw_entry in entries or []:
        entry = re.sub(r"\s+", " ", str(raw_entry).strip())
        if not entry:
            continue
        if len(entry) > CUSTOM_CATEGORY_ENTRY_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"Category entries must be {CUSTOM_CATEGORY_ENTRY_MAX} characters or fewer.",
            )
        if entry not in normalized:
            normalized.append(entry)
        if len(normalized) > CUSTOM_CATEGORY_MAX_ENTRIES:
            raise HTTPException(
                status_code=400,
                detail=f"Use {CUSTOM_CATEGORY_MAX_ENTRIES} entries or fewer.",
            )
    if not normalized:
        raise HTTPException(status_code=400, detail="Add at least one prompt idea to the category.")
    return normalized


def _validate_custom_dynamic_prompt_category(
    title: str,
    description: Optional[str],
    entries: Optional[List[str]],
    tags: Optional[List[str]],
) -> Dict[str, Any]:
    clean_title = str(title or "").strip()
    clean_description = str(description or "").strip()
    clean_entries = _normalize_custom_category_entries(entries)
    clean_tags = _normalize_community_template_tags(tags)

    if len(clean_title) < 3:
        raise HTTPException(status_code=400, detail="Category title must be at least 3 characters.")
    if len(clean_title) > CUSTOM_CATEGORY_TITLE_MAX:
        raise HTTPException(status_code=400, detail=f"Category title must be {CUSTOM_CATEGORY_TITLE_MAX} characters or fewer.")
    if len(clean_description) > CUSTOM_CATEGORY_DESCRIPTION_MAX:
        raise HTTPException(status_code=400, detail=f"Description must be {CUSTOM_CATEGORY_DESCRIPTION_MAX} characters or fewer.")

    return {
        "title": clean_title,
        "description": clean_description,
        "entries": clean_entries,
        "tags": clean_tags,
    }


def _custom_category_response(row: Any) -> Dict[str, Any]:
    entries = list(row[17] or [])
    return {
        "id": str(row[0]),
        "user_id": str(row[1]),
        "title": row[2],
        "description": row[3] or "",
        "token": row[4],
        "tags": list(row[5] or []),
        "status": row[6],
        "source_category_id": str(row[7]) if row[7] else None,
        "source_snapshot_updated_at": _isoformat(row[8]),
        "upvote_count": row[9] or 0,
        "import_count": row[10] or 0,
        "created_at": _isoformat(row[11]),
        "updated_at": _isoformat(row[12]),
        "author_display_name": row[13] or "Mobians user",
        "has_upvoted": bool(row[14]) if len(row) > 14 else False,
        "has_imported": bool(row[15]) if len(row) > 15 else False,
        "owned_category_id": str(row[16]) if len(row) > 16 and row[16] else None,
        "entries": entries,
        "examples": entries[:3],
        "item_count": len(entries),
    }


def _custom_category_select_sql(viewer_user_id: Optional[str] = None) -> str:
    viewer_uuid = "%s::uuid" if viewer_user_id else "NULL::uuid"
    return f"""
        SELECT
            c.id, c.user_id, c.title, c.description, c.token, c.tags, c.status,
            c.source_category_id, c.source_snapshot_updated_at,
            c.upvote_count, c.import_count, c.created_at, c.updated_at,
            COALESCE(
                NULLIF(CASE WHEN POSITION('@' IN COALESCE(TRIM(u.display_name), '')) = 0 THEN TRIM(u.display_name) ELSE '' END, ''),
                NULLIF(CASE WHEN POSITION('@' IN COALESCE(TRIM(u.username), '')) = 0 THEN TRIM(u.username) ELSE '' END, ''),
                'Mobians user'
            ) AS author_display_name,
            EXISTS (
                SELECT 1 FROM dynamic_prompt.custom_category_votes v
                WHERE v.category_id = c.id AND v.user_id = {viewer_uuid}
            ) AS has_upvoted,
            EXISTS (
                SELECT 1 FROM dynamic_prompt.custom_category_imports i
                WHERE i.original_category_id = c.id AND i.user_id = {viewer_uuid}
            ) AS has_imported,
            (
                SELECT i.imported_category_id FROM dynamic_prompt.custom_category_imports i
                WHERE i.original_category_id = c.id AND i.user_id = {viewer_uuid}
                LIMIT 1
            ) AS owned_category_id,
            COALESCE((
                SELECT ARRAY_AGG(ci.value ORDER BY ci.display_order, ci.value)
                FROM dynamic_prompt.custom_category_items ci
                WHERE ci.category_id = c.id AND ci.is_active = TRUE
            ), ARRAY[]::text[]) AS entries
        FROM dynamic_prompt.custom_categories c
        JOIN users u ON u.id = c.user_id
    """


async def _fetch_custom_category(
    category_id: str,
    viewer_user_id: Optional[str] = None,
    owner_user_id: Optional[str] = None,
    public_only: bool = False,
) -> Optional[Dict[str, Any]]:
    params: List[Any] = []
    viewer_placeholders = 3 if viewer_user_id else 0
    params.extend([viewer_user_id] * viewer_placeholders)
    where_clauses = ["c.id = %s"]
    params.append(category_id)
    if public_only:
        where_clauses.append("c.status = 'public'")
    if owner_user_id:
        where_clauses.append("c.user_id = %s")
        params.append(owner_user_id)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _custom_category_select_sql(viewer_user_id)
                + " WHERE " + " AND ".join(where_clauses),
                tuple(params),
            )
            row = await acur.fetchone()
    return _custom_category_response(row) if row else None


async def _replace_custom_category_items(acur: Any, category_id: str, entries: List[str]) -> None:
    await acur.execute(
        "UPDATE dynamic_prompt.custom_category_items SET is_active = FALSE, updated_at = NOW() WHERE category_id = %s",
        (category_id,),
    )
    for index, entry in enumerate(entries):
        await acur.execute(
            """
            INSERT INTO dynamic_prompt.custom_category_items (category_id, value, display_order, is_active, updated_at)
            VALUES (%s, %s, %s, TRUE, NOW())
            ON CONFLICT (category_id, value) DO UPDATE SET
                display_order = EXCLUDED.display_order,
                is_active = TRUE,
                updated_at = NOW()
            """,
            (category_id, entry, (index + 1) * 10),
        )


async def _validate_community_dynamic_prompt_template(
    title: str,
    description: Optional[str],
    template: str,
    tags: Optional[List[str]],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    clean_title = str(title or "").strip()
    clean_description = str(description or "").strip()
    clean_template = str(template or "").strip()
    clean_tags = _normalize_community_template_tags(tags)

    if len(clean_title) < 3:
        raise HTTPException(status_code=400, detail="Template title must be at least 3 characters.")
    if len(clean_title) > COMMUNITY_TEMPLATE_TITLE_MAX:
        raise HTTPException(status_code=400, detail=f"Template title must be {COMMUNITY_TEMPLATE_TITLE_MAX} characters or fewer.")
    if len(clean_description) > COMMUNITY_TEMPLATE_DESCRIPTION_MAX:
        raise HTTPException(status_code=400, detail=f"Description must be {COMMUNITY_TEMPLATE_DESCRIPTION_MAX} characters or fewer.")
    if not clean_template:
        raise HTTPException(status_code=400, detail="Dynamic prompt template is required.")

    if _extract_dynamic_prompt_template_ids(clean_template):
        raise HTTPException(status_code=400, detail="Templates can include categories and variants, but not other template tokens.")

    if not _has_dynamic_prompt_syntax(clean_template):
        raise HTTPException(status_code=400, detail="Template must include dynamic prompt syntax such as _mobian/characters_ or {a|b}.")

    assets = await _load_dynamic_prompt_assets(user_id=user_id)
    engine_template = await _prepare_dynamic_prompt_template_for_expansion(
        clean_template,
        user_id,
        assets,
        allow_template_tokens=False,
    )

    try:
        preview = preview_dynamic_prompt(
            template=engine_template,
            seed=COMMUNITY_TEMPLATE_PREVIEW_SEED,
            preview_count=DEFAULT_PREVIEW_COUNT,
            max_generations=32,
            wildcard_root_map=assets["root_map"],
        )
    except DynamicPromptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "title": clean_title,
        "description": clean_description,
        "template": clean_template,
        "tags": clean_tags,
        "preview_samples": preview.previews,
    }


def _community_template_response(row: Any, preview_samples: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "id": str(row[0]),
        "user_id": str(row[1]),
        "title": row[2],
        "description": row[3] or "",
        "template": row[4],
        "token": row[22] if len(row) > 22 and row[22] else _template_token_from_parts("user", row[2]),
        "tags": list(row[5] or []),
        "status": row[6],
        "rejection_reason": row[7],
        "source_template_id": str(row[8]) if row[8] else None,
        "source_snapshot_updated_at": _isoformat(row[9]),
        "upvote_count": row[10] or 0,
        "import_count": row[11] or 0,
        "created_at": _isoformat(row[12]),
        "updated_at": _isoformat(row[13]),
        "submitted_at": _isoformat(row[14]),
        "approved_at": _isoformat(row[15]),
        "hidden_at": _isoformat(row[16]),
        "author_display_name": row[17] or "Mobians user",
        "has_upvoted": bool(row[18]) if len(row) > 18 else False,
        "has_imported": bool(row[19]) if len(row) > 19 else False,
        "owned_template_id": str(row[20]) if len(row) > 20 and row[20] else None,
        "source_author_display_name": row[21] if len(row) > 21 else None,
        "preview_samples": preview_samples or [],
    }


def _community_template_select_sql(viewer_user_id: Optional[str] = None) -> str:
    viewer_uuid = "%s::uuid" if viewer_user_id else "NULL::uuid"
    return f"""
        SELECT
            t.id, t.user_id, t.title, t.description, t.template, t.tags, t.status,
            t.rejection_reason, t.source_template_id, t.source_snapshot_updated_at,
            t.upvote_count, t.import_count, t.created_at, t.updated_at, t.submitted_at,
            t.approved_at, t.hidden_at,
            COALESCE(
                NULLIF(CASE WHEN POSITION('@' IN COALESCE(TRIM(u.display_name), '')) = 0 THEN TRIM(u.display_name) ELSE '' END, ''),
                NULLIF(CASE WHEN POSITION('@' IN COALESCE(TRIM(u.username), '')) = 0 THEN TRIM(u.username) ELSE '' END, ''),
                'Mobians user'
            ) AS author_display_name,
            EXISTS (
                SELECT 1 FROM dynamic_prompt.template_votes v
                WHERE v.template_id = t.id AND v.user_id = {viewer_uuid}
            ) AS has_upvoted,
            EXISTS (
                SELECT 1 FROM dynamic_prompt.template_imports i
                WHERE i.original_template_id = t.id AND i.user_id = {viewer_uuid}
            ) AS has_imported,
            (
                SELECT i.imported_template_id FROM dynamic_prompt.template_imports i
                WHERE i.original_template_id = t.id AND i.user_id = {viewer_uuid}
                LIMIT 1
            ) AS owned_template_id,
            CASE
                WHEN t.source_template_id IS NULL THEN NULL
                ELSE COALESCE(
                    NULLIF(CASE WHEN POSITION('@' IN COALESCE(TRIM(source_u.display_name), '')) = 0 THEN TRIM(source_u.display_name) ELSE '' END, ''),
                    NULLIF(CASE WHEN POSITION('@' IN COALESCE(TRIM(source_u.username), '')) = 0 THEN TRIM(source_u.username) ELSE '' END, ''),
                    'Mobians user'
                )
            END AS source_author_display_name
            , t.token
        FROM dynamic_prompt.templates t
        JOIN users u ON u.id = t.user_id
        LEFT JOIN dynamic_prompt.templates source_t ON source_t.id = t.source_template_id
        LEFT JOIN users source_u ON source_u.id = source_t.user_id
    """


async def _fetch_community_template(
    template_id: str,
    viewer_user_id: Optional[str] = None,
    owner_user_id: Optional[str] = None,
    public_only: bool = False,
) -> Optional[Dict[str, Any]]:
    params: List[Any] = []
    viewer_placeholders = 3 if viewer_user_id else 0
    params.extend([viewer_user_id] * viewer_placeholders)
    where_clauses = ["t.id = %s"]
    params.append(template_id)
    if public_only:
        where_clauses.append("t.status = 'approved'")
    if owner_user_id:
        where_clauses.append("t.user_id = %s")
        params.append(owner_user_id)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _community_template_select_sql(viewer_user_id)
                + " WHERE " + " AND ".join(where_clauses),
                tuple(params),
            )
            row = await acur.fetchone()
    return _community_template_response(row) if row else None


async def _community_template_preview_samples(template: str, user_id: Optional[str] = None) -> List[str]:
    assets = await _load_dynamic_prompt_assets(user_id=user_id)
    engine_template = await _prepare_dynamic_prompt_template_for_expansion(
        template,
        user_id,
        assets,
        allow_template_tokens=False,
    )
    preview = preview_dynamic_prompt(
        template=engine_template,
        seed=COMMUNITY_TEMPLATE_PREVIEW_SEED,
        preview_count=DEFAULT_PREVIEW_COUNT,
        max_generations=32,
        wildcard_root_map=assets["root_map"],
    )
    return preview.previews


async def _award_credit_transaction(
    acur: Any,
    user_id: str,
    amount: int,
    transaction_type: str,
    description: str,
) -> Dict[str, Any]:
    await acur.execute(
        """
        UPDATE public.users
        SET credits = credits + %s
        WHERE id = %s
        RETURNING credits
        """,
        (amount, user_id),
    )
    balance_row = await acur.fetchone()
    if not balance_row:
        raise HTTPException(status_code=404, detail="User not found while awarding credits.")

    balance_after = balance_row[0]
    await acur.execute(
        """
        INSERT INTO public.credit_transactions (user_id, amount, balance_after, transaction_type, description)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (user_id, amount, balance_after, transaction_type, description),
    )
    transaction_row = await acur.fetchone()
    if not transaction_row:
        raise HTTPException(status_code=500, detail="Credit transaction was not recorded.")

    return {
        "transaction_id": transaction_row[0],
        "balance_after": balance_after,
    }


async def _award_dynamic_prompt_vote_rewards(
    acur: Any,
    content_type: str,
    content_id: str,
    content_title: str,
    creator_user_id: str,
    voter_user_id: str,
) -> Dict[str, Any]:
    if content_type not in DYNAMIC_PROMPT_VOTE_CONTENT_TYPES:
        raise ValueError(f"Unsupported dynamic prompt vote content type: {content_type}")

    await acur.execute(
        """
        INSERT INTO dynamic_prompt.vote_credit_rewards (content_type, content_id, voter_user_id, creator_user_id)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING id
        """,
        (content_type, content_id, voter_user_id, creator_user_id),
    )
    reward_row = await acur.fetchone()
    if not reward_row:
        return {
            "creator_credits_awarded": 0,
            "voter_credits_awarded": 0,
            "voter_balance_after": None,
            "voter_reward_skipped_reason": "already_rewarded",
        }

    reward_id = reward_row[0]
    for lock_user_id in sorted({str(creator_user_id), str(voter_user_id)}):
        await acur.execute(
            "SELECT id FROM public.users WHERE id = %s FOR UPDATE",
            (lock_user_id,),
        )
        if not await acur.fetchone():
            raise HTTPException(status_code=404, detail="User not found while locking credit reward rows.")

    if content_type == "template":
        creator_transaction_type = DYNAMIC_PROMPT_TEMPLATE_CREATOR_TRANSACTION_TYPE
        voter_transaction_type = DYNAMIC_PROMPT_TEMPLATE_VOTER_TRANSACTION_TYPE
        voter_daily_cap = DYNAMIC_PROMPT_TEMPLATE_VOTER_DAILY_CAP
        content_label = "template"
    else:
        creator_transaction_type = DYNAMIC_PROMPT_CATEGORY_CREATOR_TRANSACTION_TYPE
        voter_transaction_type = DYNAMIC_PROMPT_CATEGORY_VOTER_TRANSACTION_TYPE
        voter_daily_cap = DYNAMIC_PROMPT_CATEGORY_VOTER_DAILY_CAP
        content_label = "category"

    creator_award = await _award_credit_transaction(
        acur,
        creator_user_id,
        DYNAMIC_PROMPT_CREATOR_VOTE_REWARD_CREDITS,
        creator_transaction_type,
        f'Vote received on dynamic prompt {content_label} "{content_title}"',
    )

    await acur.execute(
        """
        SELECT COUNT(*)
        FROM public.credit_transactions
        WHERE user_id = %s
          AND transaction_type = %s
          AND created_at >= CURRENT_DATE
          AND created_at < CURRENT_DATE + INTERVAL '1 day'
        """,
        (voter_user_id, voter_transaction_type),
    )
    voter_reward_count_row = await acur.fetchone()
    voter_reward_count_today = int(voter_reward_count_row[0] or 0) if voter_reward_count_row else 0

    voter_award_transaction_id = None
    voter_balance_after = None
    voter_credits_awarded = 0
    voter_reward_skipped_reason = None
    if voter_reward_count_today < voter_daily_cap:
        voter_award = await _award_credit_transaction(
            acur,
            voter_user_id,
            DYNAMIC_PROMPT_VOTER_VOTE_REWARD_CREDITS,
            voter_transaction_type,
            f'Vote reward for dynamic prompt {content_label} "{content_title}"',
        )
        voter_award_transaction_id = voter_award["transaction_id"]
        voter_balance_after = voter_award["balance_after"]
        voter_credits_awarded = DYNAMIC_PROMPT_VOTER_VOTE_REWARD_CREDITS
    else:
        voter_reward_skipped_reason = "daily_cap_reached"

    await acur.execute(
        """
        UPDATE dynamic_prompt.vote_credit_rewards
        SET creator_credit_transaction_id = %s,
            voter_credit_transaction_id = %s,
            voter_reward_skipped_reason = %s
        WHERE id = %s
        """,
        (
            creator_award["transaction_id"],
            voter_award_transaction_id,
            voter_reward_skipped_reason,
            reward_id,
        ),
    )

    return {
        "creator_credits_awarded": DYNAMIC_PROMPT_CREATOR_VOTE_REWARD_CREDITS,
        "voter_credits_awarded": voter_credits_awarded,
        "voter_balance_after": voter_balance_after,
        "voter_reward_skipped_reason": voter_reward_skipped_reason,
    }


def _expand_regional_dynamic_prompts(
    regional_prompting: Optional[Dict[str, Any]],
    config: DynamicPromptingConfig,
    seed: int,
    root_map: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not regional_prompting or not regional_prompting.get("enabled"):
        return regional_prompting

    regions = regional_prompting.get("regions")
    if not isinstance(regions, list):
        return regional_prompting

    expanded_payload = json.loads(json.dumps(regional_prompting))
    expanded_regions = expanded_payload.get("regions", [])
    base_config = _dynamic_config_dict(config)
    base_config.pop("selected_preview_index", None)
    base_config["preview_count"] = 1

    for region_index, region in enumerate(expanded_regions):
        if not isinstance(region, dict):
            continue

        prompt = str(region.get("prompt", "")).strip()
        if prompt:
            region_config = dict(base_config)
            region_config["expansion_seed"] = seed + 1009 + (region_index * 17)
            region["prompt"] = expand_dynamic_prompt(
                _category_tokens_to_engine_wildcards(prompt),
                region_config,
                wildcard_root_map=root_map,
            ).expanded_prompt

        negative_prompt = str(region.get("negative_prompt", "")).strip()
        if negative_prompt:
            negative_config = dict(base_config)
            negative_config["expansion_seed"] = seed + 2009 + (region_index * 17)
            region["negative_prompt"] = expand_dynamic_prompt(
                _category_tokens_to_engine_wildcards(negative_prompt),
                negative_config,
                wildcard_root_map=root_map,
            ).expanded_prompt

    return expanded_payload


@app.get("/dynamic-prompts/library")
async def dynamic_prompt_library(user: Optional[dict] = Depends(get_current_user)):
    assets = await _load_dynamic_prompt_assets(user_id=user["user_id"] if user else None)
    return JSONResponse(content=assets["library"])


@app.post("/dynamic-prompts/preview")
async def dynamic_prompt_preview(
    request: DynamicPromptPreviewRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    try:
        assets = await _load_dynamic_prompt_assets(user_id=user["user_id"] if user else None)
        engine_template = await _prepare_dynamic_prompt_template_for_expansion(
            request.template,
            user["user_id"] if user else None,
            assets,
        )
        preview = preview_dynamic_prompt(
            template=engine_template,
            mode=request.mode,
            seed=request.seed,
            preview_count=DEFAULT_PREVIEW_COUNT,
            max_generations=request.max_generations,
            wildcard_root_map=assets["root_map"],
        )
    except DynamicPromptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        content={
            "template": str(request.template or "").strip(),
            "previews": preview.previews,
            "seed": preview.seed,
            "mode": preview.mode,
            "wildcard_set": preview.wildcard_set,
        }
    )


@app.get("/user/dynamic-prompts/categories")
async def get_user_dynamic_prompt_categories(
    status: str = "all",
    user: dict = Depends(require_auth),
):
    normalized_status = status.strip().lower()
    if normalized_status != "all" and normalized_status not in CUSTOM_CATEGORY_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid category status filter.")

    params: List[Any] = [user["user_id"], user["user_id"], user["user_id"], user["user_id"]]
    where_clauses = ["c.user_id = %s", "c.status <> 'hidden'"]
    if normalized_status != "all":
        where_clauses.append("c.status = %s")
        params.append(normalized_status)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _custom_category_select_sql(user["user_id"])
                + " WHERE " + " AND ".join(where_clauses)
                + " ORDER BY c.updated_at DESC LIMIT 100",
                tuple(params),
            )
            rows = await acur.fetchall()

    return JSONResponse(content={"categories": [_custom_category_response(row) for row in rows]})


@app.post("/user/dynamic-prompts/categories")
async def create_user_dynamic_prompt_category(
    request: DynamicPromptCustomCategoryCreate,
    user: dict = Depends(require_auth),
):
    validated = _validate_custom_dynamic_prompt_category(
        request.title,
        request.description,
        request.entries,
        request.tags,
    )
    category_id = str(uuid.uuid4())

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            category_token = await _generate_unique_custom_category_token(acur, validated["title"], category_id, user)
            await acur.execute(
                """
                INSERT INTO dynamic_prompt.custom_categories (id, user_id, title, description, token, tags, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'private')
                """,
                (
                    category_id,
                    user["user_id"],
                    validated["title"],
                    validated["description"],
                    category_token,
                    validated["tags"],
                ),
            )
            await _replace_custom_category_items(acur, category_id, validated["entries"])
            await aconn.commit()

    created = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"category": created})


@app.put("/user/dynamic-prompts/categories/{category_id}")
async def update_user_dynamic_prompt_category(
    category_id: str,
    request: DynamicPromptCustomCategoryUpdate,
    user: dict = Depends(require_auth),
):
    existing = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found.")

    validated = _validate_custom_dynamic_prompt_category(
        request.title if request.title is not None else existing["title"],
        request.description if request.description is not None else existing["description"],
        request.entries if request.entries is not None else existing["entries"],
        request.tags if request.tags is not None else existing["tags"],
    )

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            next_token = await _resolve_updated_custom_category_token(
                acur,
                user,
                category_id,
                validated["title"],
                existing.get("token"),
            )
            await acur.execute(
                """
                UPDATE dynamic_prompt.custom_categories
                SET title = %s,
                    description = %s,
                    token = %s,
                    token_aliases = %s,
                    tags = %s,
                    updated_at = NOW()
                WHERE id = %s AND user_id = %s
                """,
                (
                    validated["title"],
                    validated["description"],
                    next_token,
                    [],
                    validated["tags"],
                    category_id,
                    user["user_id"],
                ),
            )
            await _replace_custom_category_items(acur, category_id, validated["entries"])
            await aconn.commit()

    updated = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"category": updated})


@app.delete("/user/dynamic-prompts/categories/{category_id}")
async def delete_user_dynamic_prompt_category(category_id: str, user: dict = Depends(require_auth)):
    existing = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found.")
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                DELETE FROM dynamic_prompt.custom_categories
                WHERE id = %s AND user_id = %s
                """,
                (category_id, user["user_id"]),
            )
            await aconn.commit()
    return JSONResponse(content={"success": True})


@app.post("/user/dynamic-prompts/categories/{category_id}/share")
async def share_user_dynamic_prompt_category(category_id: str, user: dict = Depends(require_auth)):
    existing = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found.")
    if existing.get("status") != "public" and existing.get("source_category_id"):
        raise HTTPException(status_code=400, detail="Imported categories cannot be shared to the community.")
    if not existing["entries"]:
        raise HTTPException(status_code=400, detail="Add at least one prompt idea before sharing.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.custom_categories
                SET status = 'public', updated_at = NOW()
                WHERE id = %s AND user_id = %s AND status <> 'hidden'
                """,
                (category_id, user["user_id"]),
            )
            await aconn.commit()

    shared = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"category": shared})


@app.post("/user/dynamic-prompts/categories/{category_id}/unshare")
async def unshare_user_dynamic_prompt_category(category_id: str, user: dict = Depends(require_auth)):
    existing = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.custom_categories
                SET status = 'private', updated_at = NOW()
                WHERE id = %s AND user_id = %s AND status = 'public'
                """,
                (category_id, user["user_id"]),
            )
            await aconn.commit()

    unshared = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"category": unshared})


@app.get("/dynamic-prompts/categories")
async def list_public_dynamic_prompt_categories(
    search: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    sort: str = "top",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    user: Optional[dict] = Depends(get_current_user),
):
    normalized_sort = sort.strip().lower()
    if normalized_sort not in CUSTOM_CATEGORY_PUBLIC_SORTS:
        raise HTTPException(status_code=400, detail="Invalid category sort.")

    viewer_user_id = user["user_id"] if user else None
    params: List[Any] = []
    if viewer_user_id:
        params.extend([viewer_user_id, viewer_user_id, viewer_user_id])
    where_clauses = ["c.status = 'public'"]

    if search and search.strip():
        search_term = f"%{search.strip()}%"
        where_clauses.append("(c.title ILIKE %s OR c.description ILIKE %s)")
        params.extend([search_term, search_term])

    clean_tags = _normalize_community_template_tags(tags)
    if clean_tags:
        where_clauses.append("c.tags @> %s::text[]")
        params.append(clean_tags)

    order_by = {
        "new": "c.created_at DESC",
        "top": "c.upvote_count DESC, c.created_at DESC",
        "popular": "c.import_count DESC, c.upvote_count DESC, c.created_at DESC",
    }[normalized_sort]
    params.extend([page_size, (page - 1) * page_size])

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _custom_category_select_sql(viewer_user_id)
                + " WHERE " + " AND ".join(where_clauses)
                + f" ORDER BY {order_by} LIMIT %s OFFSET %s",
                tuple(params),
            )
            rows = await acur.fetchall()

    return JSONResponse(
        content={
            "categories": [_custom_category_response(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "sort": normalized_sort,
        }
    )


@app.post("/dynamic-prompts/categories/{category_id}/upvote")
async def upvote_dynamic_prompt_category(category_id: str, user: dict = Depends(require_auth)):
    category = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], public_only=True)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found.")
    if category["user_id"] == user["user_id"]:
        raise HTTPException(status_code=400, detail="You cannot upvote your own category.")

    vote_reward = None
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                INSERT INTO dynamic_prompt.custom_category_votes (category_id, user_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                RETURNING 1
                """,
                (category_id, user["user_id"]),
            )
            inserted = await acur.fetchone()
            if inserted:
                await acur.execute(
                    "UPDATE dynamic_prompt.custom_categories SET upvote_count = upvote_count + 1, updated_at = NOW() WHERE id = %s",
                    (category_id,),
                )
                vote_reward = await _award_dynamic_prompt_vote_rewards(
                    acur,
                    "category",
                    category_id,
                    category["title"],
                    category["user_id"],
                    user["user_id"],
                )
            await aconn.commit()

    updated = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], public_only=True)
    response_content = {"category": updated}
    if vote_reward:
        response_content["vote_reward"] = vote_reward
    return JSONResponse(content=response_content)


@app.delete("/dynamic-prompts/categories/{category_id}/upvote")
async def remove_dynamic_prompt_category_upvote(category_id: str, user: dict = Depends(require_auth)):
    category = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], public_only=True)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                DELETE FROM dynamic_prompt.custom_category_votes
                WHERE category_id = %s AND user_id = %s
                RETURNING 1
                """,
                (category_id, user["user_id"]),
            )
            deleted = await acur.fetchone()
            if deleted:
                await acur.execute(
                    """
                    UPDATE dynamic_prompt.custom_categories
                    SET upvote_count = GREATEST(upvote_count - 1, 0), updated_at = NOW()
                    WHERE id = %s
                    """,
                    (category_id,),
                )
            await aconn.commit()

    updated = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], public_only=True)
    return JSONResponse(content={"category": updated})


@app.post("/dynamic-prompts/categories/{category_id}/import")
async def import_dynamic_prompt_category(category_id: str, user: dict = Depends(require_auth)):
    category = await _fetch_custom_category(category_id, viewer_user_id=user["user_id"], public_only=True)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found.")
    if category.get("user_id") == user["user_id"]:
        raise HTTPException(status_code=400, detail="You cannot import your own category.")
    if category.get("source_category_id"):
        raise HTTPException(status_code=400, detail="Imported community categories cannot be imported again.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT imported_category_id
                FROM dynamic_prompt.custom_category_imports
                WHERE original_category_id = %s AND user_id = %s
                """,
                (category_id, user["user_id"]),
            )
            existing_import = await acur.fetchone()
            if existing_import:
                imported_id = existing_import[0]
            else:
                imported_id = str(uuid.uuid4())
                imported_token = await _generate_unique_custom_category_token(acur, category["title"], imported_id, user)
                await acur.execute(
                    """
                    INSERT INTO dynamic_prompt.custom_categories (
                        id, user_id, title, description, token, tags, status,
                        source_category_id, source_snapshot_updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'private', %s, %s)
                    """,
                    (
                        imported_id,
                        user["user_id"],
                        category["title"],
                        category["description"],
                        imported_token,
                        category["tags"],
                        category_id,
                        category["updated_at"],
                    ),
                )
                await _replace_custom_category_items(acur, imported_id, category["entries"])
                await acur.execute(
                    """
                    INSERT INTO dynamic_prompt.custom_category_imports (original_category_id, imported_category_id, user_id)
                    VALUES (%s, %s, %s)
                    """,
                    (category_id, imported_id, user["user_id"]),
                )
                await acur.execute(
                    "UPDATE dynamic_prompt.custom_categories SET import_count = import_count + 1 WHERE id = %s",
                    (category_id,),
                )
            await aconn.commit()

    imported = await _fetch_custom_category(str(imported_id), viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"category": imported})


@app.get("/user/dynamic-prompts/templates")
async def get_user_dynamic_prompt_templates(
    status: str = "all",
    user: dict = Depends(require_auth),
):
    normalized_status = status.strip().lower()
    if normalized_status != "all" and normalized_status not in COMMUNITY_TEMPLATE_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid template status filter.")

    params: List[Any] = [user["user_id"], user["user_id"], user["user_id"], user["user_id"]]
    where_clauses = ["t.user_id = %s", "t.status <> 'hidden'"]
    if normalized_status != "all":
        where_clauses.append("t.status = %s")
        params.append(normalized_status)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _community_template_select_sql(user["user_id"])
                + " WHERE " + " AND ".join(where_clauses)
                + " ORDER BY t.updated_at DESC LIMIT 100",
                tuple(params),
            )
            rows = await acur.fetchall()

    return JSONResponse(content={"templates": [_community_template_response(row) for row in rows]})


@app.post("/user/dynamic-prompts/templates")
async def create_user_dynamic_prompt_template(
    request: DynamicPromptTemplateCreate,
    user: dict = Depends(require_auth),
):
    validated = await _validate_community_dynamic_prompt_template(
        request.title,
        request.description,
        request.template,
        request.tags,
        user_id=user["user_id"],
    )

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            template_id = str(uuid.uuid4())
            template_token = await _generate_unique_template_token(acur, validated["title"], template_id, user)
            await acur.execute(
                """
                INSERT INTO dynamic_prompt.templates (id, user_id, title, description, template, token, tags, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'private')
                RETURNING id
                """,
                (
                    template_id,
                    user["user_id"],
                    validated["title"],
                    validated["description"],
                    validated["template"],
                    template_token,
                    validated["tags"],
                ),
            )
            row = await acur.fetchone()
            await aconn.commit()

    created = await _fetch_community_template(str(row[0]), viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"template": created, "preview_samples": validated["preview_samples"]})


@app.put("/user/dynamic-prompts/templates/{template_id}")
async def update_user_dynamic_prompt_template(
    template_id: str,
    request: DynamicPromptTemplateUpdate,
    user: dict = Depends(require_auth),
):
    existing = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found.")
    if existing["status"] not in COMMUNITY_TEMPLATE_MUTABLE_STATUSES:
        raise HTTPException(status_code=409, detail="This template can no longer be edited.")

    validated = await _validate_community_dynamic_prompt_template(
        request.title if request.title is not None else existing["title"],
        request.description if request.description is not None else existing["description"],
        request.template if request.template is not None else existing["template"],
        request.tags if request.tags is not None else existing["tags"],
        user_id=user["user_id"],
    )

    next_status = "approved" if existing["status"] == "approved" else "private"
    should_detach_source_template = bool(existing.get("source_template_id")) and (
        validated["title"] != existing["title"]
        or validated["description"] != existing["description"]
        or validated["template"] != existing["template"]
        or validated["tags"] != existing["tags"]
    )
    next_source_template_id = None if should_detach_source_template else existing.get("source_template_id")
    next_source_snapshot_updated_at = None if should_detach_source_template else existing.get("source_snapshot_updated_at")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            next_token = existing.get("token")
            if not next_token:
                next_token = await _generate_unique_template_token(
                    acur,
                    validated["title"],
                    template_id,
                    user,
                    exclude_template_id=template_id,
                )

            if should_detach_source_template:
                await acur.execute(
                    """
                    DELETE FROM dynamic_prompt.template_imports
                    WHERE imported_template_id = %s AND user_id = %s
                    RETURNING original_template_id
                    """,
                    (template_id, user["user_id"]),
                )
                detached_import = await acur.fetchone()
                if detached_import and detached_import[0]:
                    await acur.execute(
                        """
                        UPDATE dynamic_prompt.templates
                        SET import_count = GREATEST(import_count - 1, 0), updated_at = NOW()
                        WHERE id = %s
                        """,
                        (detached_import[0],),
                    )

            await acur.execute(
                """
                UPDATE dynamic_prompt.templates
                SET title = %s,
                    description = %s,
                    template = %s,
                    token = %s,
                    tags = %s,
                    status = %s,
                    rejection_reason = NULL,
                    source_template_id = %s,
                    source_snapshot_updated_at = %s,
                    submitted_at = NULL,
                    approved_at = CASE WHEN %s = 'approved' THEN COALESCE(approved_at, NOW()) ELSE NULL END,
                    updated_at = NOW()
                WHERE id = %s AND user_id = %s
                """,
                (
                    validated["title"],
                    validated["description"],
                    validated["template"],
                    next_token,
                    validated["tags"],
                    next_status,
                    next_source_template_id,
                    next_source_snapshot_updated_at,
                    next_status,
                    template_id,
                    user["user_id"],
                ),
            )
            await aconn.commit()

    updated = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"template": updated, "preview_samples": validated["preview_samples"]})


@app.delete("/user/dynamic-prompts/templates/{template_id}")
async def delete_user_dynamic_prompt_template(template_id: str, user: dict = Depends(require_auth)):
    existing = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found.")
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                DELETE FROM dynamic_prompt.templates
                WHERE id = %s AND user_id = %s
                """,
                (template_id, user["user_id"]),
            )
            await aconn.commit()
    return JSONResponse(content={"success": True})


@app.post("/user/dynamic-prompts/templates/{template_id}/share")
async def share_user_dynamic_prompt_template(template_id: str, user: dict = Depends(require_auth)):
    existing = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found.")
    if existing["status"] not in COMMUNITY_TEMPLATE_MUTABLE_STATUSES:
        raise HTTPException(status_code=409, detail="This template can no longer be shared.")
    if existing["status"] != "approved" and existing.get("source_template_id"):
        raise HTTPException(status_code=400, detail="Imported templates cannot be shared to the community.")

    preview_samples = (await _validate_community_dynamic_prompt_template(
        existing["title"],
        existing["description"],
        existing["template"],
        existing["tags"],
        user_id=user["user_id"],
    ))["preview_samples"]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.templates
                SET status = 'approved', rejection_reason = NULL, submitted_at = NULL, approved_at = COALESCE(approved_at, NOW()), updated_at = NOW()
                WHERE id = %s AND user_id = %s AND status <> 'hidden'
                """,
                (template_id, user["user_id"]),
            )
            await aconn.commit()

    shared = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"template": shared, "preview_samples": preview_samples})


@app.post("/user/dynamic-prompts/templates/{template_id}/unshare")
async def unshare_user_dynamic_prompt_template(template_id: str, user: dict = Depends(require_auth)):
    existing = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.templates
                SET status = 'private', rejection_reason = NULL, submitted_at = NULL, approved_at = NULL, updated_at = NOW()
                WHERE id = %s AND user_id = %s AND status = 'approved'
                """,
                (template_id, user["user_id"]),
            )
            await aconn.commit()

    unshared = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"template": unshared})


@app.post("/user/dynamic-prompts/templates/{template_id}/submit")
async def submit_user_dynamic_prompt_template(template_id: str, user: dict = Depends(require_auth)):
    return await share_user_dynamic_prompt_template(template_id, user)


@app.get("/dynamic-prompts/templates")
async def list_public_dynamic_prompt_templates(
    search: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    sort: str = "top",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    user: Optional[dict] = Depends(get_current_user),
):
    normalized_sort = sort.strip().lower()
    if normalized_sort not in COMMUNITY_TEMPLATE_PUBLIC_SORTS:
        raise HTTPException(status_code=400, detail="Invalid template sort.")

    viewer_user_id = user["user_id"] if user else None
    params: List[Any] = []
    if viewer_user_id:
        params.extend([viewer_user_id, viewer_user_id, viewer_user_id])
    where_clauses = ["t.status = 'approved'"]

    if search and search.strip():
        params.append(f"%{search.strip()}%")
        where_clauses.append("(t.title ILIKE %s OR t.description ILIKE %s OR t.template ILIKE %s)")
        params.extend([params[-1], params[-1]])

    clean_tags = _normalize_community_template_tags(tags)
    if clean_tags:
        where_clauses.append("t.tags @> %s::text[]")
        params.append(clean_tags)

    order_by = {
        "new": "t.approved_at DESC NULLS LAST, t.created_at DESC",
        "top": "t.upvote_count DESC, t.approved_at DESC NULLS LAST",
        "popular": "t.import_count DESC, t.upvote_count DESC, t.approved_at DESC NULLS LAST",
    }[normalized_sort]
    params.extend([page_size, (page - 1) * page_size])

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _community_template_select_sql(viewer_user_id)
                + " WHERE " + " AND ".join(where_clauses)
                + f" ORDER BY {order_by} LIMIT %s OFFSET %s",
                tuple(params),
            )
            rows = await acur.fetchall()

    return JSONResponse(
        content={
            "templates": [_community_template_response(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "sort": normalized_sort,
        }
    )


@app.get("/dynamic-prompts/templates/{template_id}")
async def get_public_dynamic_prompt_template(
    template_id: str,
    user: Optional[dict] = Depends(get_current_user),
):
    viewer_user_id = user["user_id"] if user else None
    template = await _fetch_community_template(template_id, viewer_user_id=viewer_user_id, public_only=True)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")
    template["preview_samples"] = await _community_template_preview_samples(template["template"], viewer_user_id)
    return JSONResponse(content={"template": template})


@app.post("/dynamic-prompts/templates/{template_id}/upvote")
async def upvote_dynamic_prompt_template(template_id: str, user: dict = Depends(require_auth)):
    template = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], public_only=True)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")
    if template["user_id"] == user["user_id"]:
        raise HTTPException(status_code=400, detail="You cannot upvote your own template.")

    vote_reward = None
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                INSERT INTO dynamic_prompt.template_votes (template_id, user_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                RETURNING 1
                """,
                (template_id, user["user_id"]),
            )
            inserted = await acur.fetchone()
            if inserted:
                await acur.execute(
                    "UPDATE dynamic_prompt.templates SET upvote_count = upvote_count + 1, updated_at = NOW() WHERE id = %s",
                    (template_id,),
                )
                vote_reward = await _award_dynamic_prompt_vote_rewards(
                    acur,
                    "template",
                    template_id,
                    template["title"],
                    template["user_id"],
                    user["user_id"],
                )
            await aconn.commit()

    updated = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], public_only=True)
    response_content = {"template": updated}
    if vote_reward:
        response_content["vote_reward"] = vote_reward
    return JSONResponse(content=response_content)


@app.delete("/dynamic-prompts/templates/{template_id}/upvote")
async def remove_dynamic_prompt_template_upvote(template_id: str, user: dict = Depends(require_auth)):
    template = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], public_only=True)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                DELETE FROM dynamic_prompt.template_votes
                WHERE template_id = %s AND user_id = %s
                RETURNING 1
                """,
                (template_id, user["user_id"]),
            )
            deleted = await acur.fetchone()
            if deleted:
                await acur.execute(
                    """
                    UPDATE dynamic_prompt.templates
                    SET upvote_count = GREATEST(upvote_count - 1, 0), updated_at = NOW()
                    WHERE id = %s
                    """,
                    (template_id,),
                )
            await aconn.commit()

    updated = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], public_only=True)
    return JSONResponse(content={"template": updated})


@app.post("/dynamic-prompts/templates/{template_id}/import")
async def import_dynamic_prompt_template(template_id: str, user: dict = Depends(require_auth)):
    template = await _fetch_community_template(template_id, viewer_user_id=user["user_id"], public_only=True)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")
    if template.get("user_id") == user["user_id"]:
        raise HTTPException(status_code=400, detail="You cannot import your own template.")
    if template.get("source_template_id"):
        raise HTTPException(status_code=400, detail="Imported community templates cannot be imported again.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT imported_template_id
                FROM dynamic_prompt.template_imports
                WHERE original_template_id = %s AND user_id = %s
                """,
                (template_id, user["user_id"]),
            )
            existing_import = await acur.fetchone()
            if existing_import:
                imported_id = existing_import[0]
            else:
                imported_id = str(uuid.uuid4())
                imported_token = await _generate_unique_template_token(acur, template["title"], imported_id, user)
                await acur.execute(
                    """
                    INSERT INTO dynamic_prompt.templates (
                        id, user_id, title, description, template, token, tags, status,
                        source_template_id, source_snapshot_updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'private', %s, %s)
                    RETURNING id
                    """,
                    (
                        imported_id,
                        user["user_id"],
                        template["title"],
                        template["description"],
                        template["template"],
                        imported_token,
                        template["tags"],
                        template_id,
                        template["updated_at"],
                    ),
                )
                imported_row = await acur.fetchone()
                imported_id = imported_row[0]
                await acur.execute(
                    """
                    INSERT INTO dynamic_prompt.template_imports (original_template_id, imported_template_id, user_id)
                    VALUES (%s, %s, %s)
                    """,
                    (template_id, imported_id, user["user_id"]),
                )
                await acur.execute(
                    "UPDATE dynamic_prompt.templates SET import_count = import_count + 1 WHERE id = %s",
                    (template_id,),
                )
            await aconn.commit()

    imported = await _fetch_community_template(str(imported_id), viewer_user_id=user["user_id"], owner_user_id=user["user_id"])
    return JSONResponse(content={"template": imported})


@app.post("/submit_job/")
async def submit_job(
    job_data: JobData,
    background_tasks: BackgroundTasks,
    user: Optional[dict] = Depends(get_current_user),
):
    # Ensure model is always valid even if clients send stale/renamed ids.
    job_data.model = normalize_model_id(job_data.model)

    # CRITICAL: Validate that img2img/inpainting/upscale jobs have required image data.
    # This prevents a frontend bug where job_type is set but image data is missing.
    if job_data.job_type in ("img2img", "inpainting", "upscale"):
        if not job_data.image or not job_data.image.strip():
            logging.warning(
                f"Received {job_data.job_type} job without image data. "
                "This is likely a frontend bug - falling back to txt2img."
            )
            job_data.job_type = "txt2img"
            job_data.image = None
            job_data.mask_image = None
    
    if job_data.job_type == "inpainting":
        if not job_data.mask_image or not job_data.mask_image.strip():
            logging.warning(
                "Received inpainting job without mask data. "
                "Falling back to img2img."
            )
            job_data.job_type = "img2img" if job_data.image else "txt2img"
            job_data.mask_image = None

    # Clear color_inpaint if no mask is present.
    # Prevents filter_image() from trying to process a null mask when
    # the frontend accidentally sends color_inpaint=True without mask data.
    if job_data.color_inpaint and (not job_data.mask_image or not job_data.mask_image.strip()):
        logging.warning(
            "Clearing color_inpaint flag: no mask_image provided. "
            "This is likely a stale frontend state."
        )
        job_data.color_inpaint = None

    # Determine queue type and credit cost
    queue_type = job_data.queue_type or "free"
    credit_cost = 0
    user_id = None

    is_upscale_job = (job_data.job_type == "upscale")
    is_hires_job = (job_data.job_type == HIRES_JOB_TYPE)
    if is_upscale_job or is_hires_job:
        # Expensive jobs: require priority queue + authentication
        queue_type = "priority"
    
    # If user wants priority queue, they must be authenticated and have credits
    if queue_type == "priority":
        if not user:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Authentication required. Please log in to upscale images."
                    if is_upscale_job
                    else (
                        "Authentication required. Please log in to use Hi-Res generation."
                        if is_hires_job
                        else "Authentication required for priority queue. Please log in or use the free queue."
                    )
                ),
            )
        
        user_id = user["user_id"]
        credit_cost = (
            get_upscale_credit_cost(job_data.model, job_data.loras)
            if is_upscale_job
            else (
                get_hires_credit_cost(job_data.model, job_data.loras)
                if is_hires_job
                else get_credit_cost(job_data.model, job_data.loras)
            )
        )
        
        if user["credits"] < credit_cost:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient credits. You need {credit_cost} credits but have {user['credits']}. Use the free queue or purchase more credits."
            )
    elif user:
        # Free queue but user is logged in - track the user_id anyway
        user_id = user["user_id"]
    
    # Check if FastPassCode is valid and non-expired (legacy system - overrides queue_type)
    # fast_pass_enabled is used by the queue view to determine priority order
    fast_pass_enabled = (queue_type == "priority")
    if job_data.fast_pass_code:
        try:
            is_valid = await validate_fastpass(
                job_data.fast_pass_code, background_tasks
            )
            if is_valid:
                fast_pass_enabled = True
                queue_type = "priority"  # FastPass gives priority
                if not (is_upscale_job or is_hires_job):
                    credit_cost = 0  # FastPass is free
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or expired FastPassCode. Please fix/remove the FastPassCode and try again.",
                )
        except HTTPException as e:
            raise e
        except Exception as e:
            logging.error(
                "Error occurred while validating FastPassCode (DB might be down)"
            )
            logging.error(str(e))
            raise HTTPException(
                status_code=500,
                detail="An error occurred while validating the FastPassCode. Please try again later.",
            )

    dynamic_prompt_template: Optional[str] = None
    dynamic_prompt_engine_template: Optional[str] = None
    dynamic_expanded_prompt: Optional[str] = None
    dynamic_prompt_candidate = str(
        (job_data.dynamic_prompting.template if job_data.dynamic_prompting and job_data.dynamic_prompting.template else job_data.prompt)
        or ""
    )
    should_try_dynamic_prompt = bool(
        (job_data.dynamic_prompting and job_data.dynamic_prompting.enabled)
        or _has_dynamic_prompt_syntax(dynamic_prompt_candidate)
    )
    allowed_wildcards: set[str] = set()
    if should_try_dynamic_prompt:
        dynamic_prompt_assets = await _load_dynamic_prompt_assets(user_id=user["user_id"] if user else None)
        allowed_wildcards = _allowed_dynamic_prompt_wildcard_ids(dynamic_prompt_assets)
        resolved_dynamic_prompt = _resolve_dynamic_prompt_request(
            job_data.prompt,
            job_data.dynamic_prompting,
            allowed_wildcards,
        )
    else:
        dynamic_prompt_assets = None
        resolved_dynamic_prompt = None

    if resolved_dynamic_prompt and dynamic_prompt_assets:
        dynamic_prompt_template, dynamic_prompt_config = resolved_dynamic_prompt
        dynamic_prompt_engine_template = await _prepare_dynamic_prompt_template_for_expansion(
            dynamic_prompt_template,
            user["user_id"] if user else None,
            dynamic_prompt_assets,
        )
        job_data.dynamic_prompting = dynamic_prompt_config
        try:
            expansion = expand_dynamic_prompt(
                dynamic_prompt_engine_template,
                _dynamic_config_dict(dynamic_prompt_config),
                fallback_seed=job_data.seed,
                wildcard_root_map=dynamic_prompt_assets["root_map"],
            )
            job_data.prompt = expansion.expanded_prompt
            dynamic_expanded_prompt = expansion.expanded_prompt
            dynamic_prompt_config.expansion_seed = expansion.seed
            dynamic_prompt_config.selected_preview_index = expansion.selected_preview_index
            job_data.regional_prompting = _expand_regional_dynamic_prompts(
                job_data.regional_prompting,
                dynamic_prompt_config,
                expansion.seed,
                dynamic_prompt_assets["root_map"],
            )
        except DynamicPromptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Filter out prompts
    job_data.prompt, job_data.negative_prompt = await promptFilter(job_data)
    job_data.negative_prompt = await fortify_default_negative(job_data.negative_prompt)
    if dynamic_prompt_template:
        dynamic_expanded_prompt = job_data.prompt

    # Create an instance of ImageRequestModel
    image_request_data = ImageRequestModel(
        **job_data.dict(), 
        fast_pass_enabled=fast_pass_enabled,
        user_id=user_id,
        credit_cost=credit_cost
    )

    regional_prompt_payload: Optional[str] = None
    if image_request_data.regional_prompting and image_request_data.regional_prompting.get("enabled"):
        regional_prompt_payload = "__regional_prompting__:" + json.dumps(
            image_request_data.regional_prompting
        )

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # If priority queue with credits, deduct first (atomic with job creation)
            if queue_type == "priority" and credit_cost > 0 and user_id:
                # Deduct credits
                await acur.execute(
                    "SELECT * FROM deduct_credits(%s, %s, %s, NULL, %s)",
                    (user_id, credit_cost, f"generation_{MODEL_BASE_TYPES.get(job_data.model, 'SD 1.5').lower().replace(' ', '')}", 
                     f"Priority generation - {job_data.model}")
                )
                deduct_result = await acur.fetchone()
                if not deduct_result or not deduct_result[0]:
                    raise HTTPException(
                        status_code=402,
                        detail=f"Failed to deduct credits: {deduct_result[2] if deduct_result else 'Unknown error'}"
                    )
                new_balance = deduct_result[1]
            else:
                new_balance = user["credits"] if user else None
            await acur.execute(
                """
                INSERT INTO generation_queue (
                    id, status, assigned_gpu, prompt, prompt_template, image, image_UUID, mask_image,
                    color_inpaint, control_image, scheduler, steps, negative_prompt,
                    width, height, guidance_scale, seed, batch_size, strength,
                    job_type, model, fast_pass_code, rating, enable_upscale, fast_pass_enabled, 
                    is_dev_job, loras, lossy_images, user_id, queue_type, credit_cost
                ) VALUES (
                    gen_random_uuid(), 'pending', NULL,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id;
            """,
                (
                    image_request_data.prompt,
                    dynamic_prompt_template,
                    image_request_data.image,
                    image_request_data.image_UUID,
                    image_request_data.mask_image,
                    image_request_data.color_inpaint,
                    regional_prompt_payload or image_request_data.control_image,
                    image_request_data.scheduler,
                    image_request_data.steps,
                    image_request_data.negative_prompt,
                    image_request_data.width,
                    image_request_data.height,
                    image_request_data.guidance_scale,
                    image_request_data.seed,
                    image_request_data.batch_size,
                    image_request_data.strength,
                    image_request_data.job_type,
                    image_request_data.model,
                    image_request_data.fast_pass_code,
                    image_request_data.rating,
                    image_request_data.enable_upscale,
                    fast_pass_enabled,
                    image_request_data.is_dev_job,
                    json.dumps(image_request_data.loras),  # Convert loras to JSON
                    image_request_data.lossy_images,
                    user_id,
                    queue_type,
                    credit_cost,
                ),
            )
            job_id = await acur.fetchone()
            
            # Update the credit transaction with job_id if credits were deducted
            if queue_type == "priority" and credit_cost > 0 and user_id:
                await acur.execute(
                    """
                    UPDATE credit_transactions 
                    SET job_id = %s 
                    WHERE id = (
                        SELECT id FROM credit_transactions 
                        WHERE user_id = %s AND job_id IS NULL 
                        ORDER BY created_at DESC LIMIT 1
                    )
                    """,
                    (job_id[0], user_id)
                )

    response_data: Dict[str, Any] = {"job_id": str(job_id[0]), "queue_type": queue_type}
    if dynamic_expanded_prompt:
        response_data["expanded_prompt"] = dynamic_expanded_prompt
        if dynamic_prompt_template:
            response_data["prompt_template"] = dynamic_prompt_template
    if credit_cost > 0:
        response_data["credits_used"] = credit_cost
        response_data["credits_remaining"] = new_balance
    
    return JSONResponse(content=response_data)

@app.get("/search_civitAi_loras_by_query/{query}")
async def search_civitAi_loras_by_query(query: str, show_nsfw: bool = False):
    civiAi_url = "https://civitai.com/api/v1/models"

    headers: Dict[str, str] = {}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    params = {
        # CivitAI expects `types=LORA` in the query string.
        # Passing a list can serialize to repeated params (`types=LORA&types=...`) which the API may reject.
        "types": "LORA",
        "sort": "Highest Rated", 
        "period": "AllTime",
        "limit": 30,
        # "tag": "hentai", 
        "query": query,
        # "primaryFileOnly": True,
        # If you want safer-only results, set nsfw=False. Omitting keeps default behavior.
        # "allowNoCredit": True,
        # "hidden": False,
    }

    # CivitAI defaults to SFW-only when `nsfw` is omitted, so always send it explicitly.
    # aiohttp/yarl reject bool query param values; send as string.
    params["nsfw"] = "true" if show_nsfw else "false"

    async with session.get(civiAi_url, headers=headers, params=params) as resp:
        if resp.status != 200:
            upstream_body = await resp.text()
            raise HTTPException(
                status_code=resp.status,
                detail=(
                    "Error in CivitAi search, ensure the query is correct. "
                    + (f"Upstream: {upstream_body[:200]}" if upstream_body else "")
                ).strip(),
            )
        data = await resp.json()

    # We need to filter the data since it returns main pages that could contain several loras
    loras = []
    data = data['items']
    for lora in data:
        lora_page_info = {
            'model_page_id': lora.get('id'),
            'name': lora.get('name'),
            'description': lora.get('description'),
            'minor': lora.get('minor'),
            'poi': lora.get('poi'),
            'nsfw': lora.get('nsfw'),
            'creator': lora.get('creator'),
            'tags': lora.get('tags'),
        }

        for model in lora['modelVersions']:
            lora_info = {
                'model_version_id': model.get('id'),
                'model_name': model.get('name'),
                'base_model': model.get('baseModel'), 
                'published_at': model.get('publishedAt'),
                'nsfw_level': model.get('nsfwLevel'),
                'description': model.get('description'),
                'trained_words': model.get('trainedWords'),
                'stats': model.get('stats'),
                'files': model.get('files'),
                'images': model.get('images'),
                'donwload_url': model.get('downloadUrl'),
            }

            # Only surface LoRAs that match the model families supported in the app.
            if not is_supported_lora_base_model(lora_info['base_model']):
                continue

            # add the lora_info to the lora_page_info
            lora_page_info['model'] = lora_info

            loras.append({**lora_page_info, **lora_info})
        pass

    return JSONResponse(content=loras)

@app.get("/search_civitAi_loras_by_id/{id}")
async def search_civitAi_loras_by_id(id: str, show_nsfw: bool = False):
    civiAi_url = f"https://civitai.com/api/v1/models/{id}"

    headers: Dict[str, str] = {}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"

    async with session.get(civiAi_url, headers=headers) as resp:
        if resp.status != 200:
            raise HTTPException(
                status_code=resp.status, detail="Error in CivitAi search, ensure the model id is correct"
            )
        data = await resp.json()

    loras = []
    if 'modelVersions' in data:
        if data.get('nsfw') and not show_nsfw:
            raise HTTPException(
                status_code=422,
                detail="Model is marked NSFW. Enable 'Show NSFW results' to view it.",
            )
        if data.get('type') != 'LORA' and data.get('type') != 'LoCon':
            raise HTTPException(
                status_code=422, detail="Model is not a LORA, please ensure the model you are searching for is a LORA"
            )

        lora_page_info = {
            'model_page_id': data.get('id'),
            'name': data.get('name'),
            'description': data.get('description'),
            'minor': data.get('minor'),
            'poi': data.get('poi'),
            'nsfw': data.get('nsfw'),
            'creator': data.get('creator'),
            'tags': data.get('tags'),
        }

        for model in data['modelVersions']:
            lora_info = {
                'model_version_id': model.get('id'),
                'model_name': model.get('name'),
                'base_model': model.get('baseModel'), 
                'published_at': model.get('publishedAt'),
                'nsfw_level': model.get('nsfwLevel'),
                'description': model.get('description'),
                'trained_words': model.get('trainedWords'),
                'stats': model.get('stats'),
                'files': model.get('files'),
                'images': model.get('images'),
                'donwload_url': model.get('downloadUrl'),
            }

            # Only surface LoRAs that match the model families supported in the app.
            if not is_supported_lora_base_model(lora_info['base_model']):
                continue

            # add the lora_info to the lora_page_info
            lora_page_info['model'] = lora_info

            loras.append({**lora_page_info, **lora_info})
    else:
        # error
        raise HTTPException(
                status_code=resp.status, detail="No model found"
            )

    return JSONResponse(content=loras)

@app.get("/search_civitAi_loras_by_user/{username}")
async def search_civitAi_loras_by_user(username: str, show_nsfw: bool = False):
    civiAi_url = "https://civitai.com/api/v1/models"

    headers: Dict[str, str] = {}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"

    params: Dict[str, Any] = {
        "username": username,
        "types": "LORA",
        "limit": 30,
    }

    # CivitAI defaults to SFW-only when `nsfw` is omitted, so always send it explicitly.
    # aiohttp/yarl reject bool query param values; send as string.
    params["nsfw"] = "true" if show_nsfw else "false"

    async with session.get(civiAi_url, headers=headers, params=params) as resp:
        if resp.status != 200:
            raise HTTPException(
                status_code=resp.status, detail="Error in CivitAi search, ensure the username is correct"
            )
        data = await resp.json()

    # We need to filter the data since it returns main pages that could contain several loras
    loras = []
    data = data['items']
    for lora in data:
        if lora.get('type') != 'LORA' and lora.get('type') != 'LoCon':
            continue

        lora_page_info = {
            'model_page_id': lora.get('id'),
            'name': lora.get('name'),
            'description': lora.get('description'),
            'minor': lora.get('minor'),
            'poi': lora.get('poi'),
            'nsfw': lora.get('nsfw'),
            'creator': lora.get('creator'),
            'tags': lora.get('tags'),
        }

        for model in lora['modelVersions']:
            lora_info = {
                'model_version_id': model.get('id'),
                'model_name': model.get('name'),
                'base_model': model.get('baseModel'), 
                'published_at': model.get('publishedAt'),
                'nsfw_level': model.get('nsfwLevel'),
                'description': model.get('description'),
                'trained_words': model.get('trainedWords'),
                'stats': model.get('stats'),
                'files': model.get('files'),
                'images': model.get('images'),
                'donwload_url': model.get('downloadUrl'),
            }

            # Only surface LoRAs that match the model families supported in the app.
            if not is_supported_lora_base_model(lora_info['base_model']):
                continue

            # add the lora_info to the lora_page_info
            lora_page_info['model'] = lora_info

            loras.append({**lora_page_info, **lora_info})
        pass

    return JSONResponse(content=loras)


class addLoraSuggestion(BaseModel):
    lora_version_id: int
    name: str
    version: str
    status: str
    requestor: Optional[str] = None
    is_nsfw: bool
    is_minor: bool
    preview_image: str
    base_model: Optional[str] = None

@app.post("/add_lora_suggestion/")
async def add_lora_suggestion(lora_data: addLoraSuggestion, user: dict = Depends(require_auth)):
    requestor_candidates = get_lora_requestor_candidates(user)
    requestor = get_primary_lora_requestor(user)

    # First we check if the lora is in the database already
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT * FROM lora_metadata
                WHERE version_id = %s
                """,
                (lora_data.lora_version_id,),
            )
            row = await acur.fetchone()

    if row:
        return JSONResponse(content={"status": "error", "detail": "This lora already exists! Check out the loras tab :), if this is a mistake, report it on the discord!"}, status_code=400)

    # Enforce per-user active suggestion cap
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT COUNT(*)
                FROM lora_suggestions
                WHERE requestor = ANY(%s)
                  AND status = 'pending'
                """,
                (requestor_candidates,)
            )
            count_row = await acur.fetchone()
            active_count = count_row[0] if count_row else 0

    if active_count >= LORA_SUGGESTION_LIMIT:
        return JSONResponse(
            content={
                "status": "error",
                "detail": (
                    f"Suggestion limit reached ({LORA_SUGGESTION_LIMIT} active suggestions). "
                    "Please wait for your existing suggestions to be processed."
                ),
            },
            status_code=429,
        )

    # Try to insert the suggestion into the database
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    """
                    INSERT INTO lora_suggestions (version_id, name, version, status, requestor, is_nsfw, is_minor, preview_image, base_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        lora_data.lora_version_id,
                        lora_data.name,
                        lora_data.version,
                        lora_data.status,
                        requestor,
                        lora_data.is_nsfw,
                        lora_data.is_minor,
                        lora_data.preview_image,
                        lora_data.base_model,
                    ),
                )
        return JSONResponse(content={"status": "success"})
    except errors.UniqueViolation:
        # Resolve duplicate requests by checking the current status and re-queuing when safe
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    """
                    SELECT status,
                           last_updated_date,
                           COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day') AS rerequest_available_at,
                           GREATEST(
                               0,
                               CEIL(EXTRACT(EPOCH FROM ((COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day')) - NOW())))::INTEGER
                           ) AS cooldown_seconds_remaining
                    FROM lora_suggestions
                    WHERE version_id = %s
                    """,
                    (LORA_REREQUEST_COOLDOWN_DAYS, LORA_REREQUEST_COOLDOWN_DAYS, lora_data.lora_version_id),
                )
                row = await acur.fetchone()

        if not row:
            return JSONResponse(
                content={
                    "status": "error",
                    "detail": "Unable to verify existing LoRA suggestion. Please try again later.",
                },
                status_code=400,
            )

        existing_status = (row[0] or "").strip().lower()
        rerequest_available_at = row[2]
        cooldown_seconds_remaining = int(row[3] or 0)
        # Allow re-queueing when the previous attempt already finished processing
        immutable_statuses = {"pending", "approved", "downloading"}
        if existing_status in immutable_statuses:
            detail = "A suggestion for this LoRA is already pending approval. Please be patient as we review it."
            if existing_status == "approved":
                detail = "This LoRA has already been approved and is queued for download."
            elif existing_status == "downloading":
                detail = "This LoRA is currently downloading. Please wait for it to finish."
            return JSONResponse(content={"status": "error", "detail": detail}, status_code=400)

        if existing_status == "rejected" and cooldown_seconds_remaining > 0:
            days_remaining = max(1, math.ceil(cooldown_seconds_remaining / 86400))
            day_label = "day" if days_remaining == 1 else "days"
            return JSONResponse(
                content=jsonable_encoder({
                    "status": "error",
                    "detail": (
                        f"This LoRA was rejected recently. You can re-request it in "
                        f"{days_remaining} {day_label}."
                    ),
                    "cooldown_seconds_remaining": cooldown_seconds_remaining,
                    "rerequest_available_at": rerequest_available_at,
                }),
                status_code=429,
            )

        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    """
                    UPDATE lora_suggestions
                    SET name = %s,
                        version = %s,
                        status = 'pending',
                        requestor = %s,
                        is_nsfw = %s,
                        is_minor = %s,
                        preview_image = %s,
                        base_model = %s,
                        error_message = NULL,
                        last_updated_date = NOW(),
                        created_at = NOW()
                    WHERE version_id = %s
                    """,
                    (
                        lora_data.name,
                        lora_data.version,
                        requestor,
                        lora_data.is_nsfw,
                        lora_data.is_minor,
                        lora_data.preview_image,
                        lora_data.base_model,
                        lora_data.lora_version_id,
                    ),
                )

        return JSONResponse(content={"status": "success", "detail": "LoRA suggestion re-queued"})

def decode_base64_to_image(base64_str):
    # Convert base64 string to image
    try:
        image = Image.open(io.BytesIO(base64.b64decode(base64_str.split(",", 1)[1])))
    except:
        image = Image.open(io.BytesIO(base64.b64decode(base64_str.split(",", 1)[0])))

    return image


async def get_pending_queue_position(acur, job_id: str, fast_pass_enabled: bool, create_date) -> Optional[int]:
    if create_date is None:
        return None

    # Count only pending jobs ahead of the current one instead of sorting the full queue view.
    await acur.execute(
        """
        SELECT COUNT(*) + 1
        FROM generation_queue pending
        WHERE pending.status = 'pending'
          AND (
                COALESCE(pending.fast_pass_enabled, FALSE) > %s
                OR (
                    COALESCE(pending.fast_pass_enabled, FALSE) = %s
                    AND (
                        pending.create_date < %s
                        OR (pending.create_date = %s AND pending.id < %s)
                    )
                )
              )
        """,
        (fast_pass_enabled, fast_pass_enabled, create_date, create_date, job_id),
    )
    row = await acur.fetchone()
    return row[0] if row else None


async def get_generation_job_details(job_id: str) -> Optional[dict[str, Any]]:
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT id, status, create_date, COALESCE(fast_pass_enabled, FALSE),
                      finished_images, prompt, prompt_template, negative_prompt, seed, guidance_scale,
                       job_type, model, error_message, loras, lossy_images,
                       user_id, credit_cost, COALESCE(refunded, FALSE), control_image
                FROM generation_queue
                WHERE id = %s
                """,
                (job_id,),
            )
            row = await acur.fetchone()

            if not row:
                return None

            queue_position = None
            if row[1] == "pending":
                queue_position = await get_pending_queue_position(acur, str(row[0]), row[3], row[2])

    return {
        "id": str(row[0]),
        "status": row[1],
        "queue_position": queue_position,
        "finished_images": row[4],
        "prompt": row[5],
        "prompt_template": row[6],
        "negative_prompt": row[7],
        "seed": row[8],
        "guidance_scale": row[9],
        "job_type": row[10],
        "model": row[11],
        "error_message": row[12],
        "loras": row[13],
        "lossy_images": row[14],
        "user_id": row[15],
        "credit_cost": row[16],
        "refunded": row[17],
        "control_image": row[18],
    }


def parse_regional_prompting(raw_control_image: Optional[str]) -> Optional[dict[str, Any]]:
    if raw_control_image and isinstance(raw_control_image, str) and raw_control_image.startswith('__regional_prompting__:'):
        try:
            return json.loads(raw_control_image[len('__regional_prompting__:'):])
        except (json.JSONDecodeError, TypeError):
            return None
    return None


async def increment_fastpass_use_count(fast_pass_code: str):
    # Time the function
    start_time = time.time()
    logging.info("Incrementing FastPassCode use count")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE fastpass_new
                SET uses = uses + 1
                WHERE fastpass_code = %s
                """,
                (fast_pass_code,),
            )

    # Time the function
    end_time = time.time()
    logging.info("Time elapsed: " + str(end_time - start_time))


async def set_fastpass_expiration_date(fast_pass_code: str, days_from_today: int):
    # Time the function
    start_time = time.time()
    logging.info("Setting FastPassCode expiration date")

    try:
        # Add days to current date
        expiration_date = datetime.now() + timedelta(days=days_from_today)

        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    """
                    UPDATE fastpass_new
                    SET expiration_date = %s
                    WHERE fastpass_code = %s
                    """,
                    (expiration_date, fast_pass_code),
                )

        # Log the time elapsed
        end_time = time.time()
        logging.info("Time elapsed: {:.2f} seconds".format(end_time - start_time))

    except Exception as e:
        logging.error("Error setting expiration date: %s", e)


async def validate_fastpass(
    fast_pass_code: str, background_tasks: BackgroundTasks
) -> bool:
    # if fast_pass_code in fastpass_cache:
    #     expiration_date = fastpass_cache[fast_pass_code]
    #     if expiration_date is None or expiration_date >= datetime.now():
    #         background_tasks.add_task(increment_fastpass_use_count, fast_pass_code)
    #         return True
    #     else:
    #         return False
    # else:
    # FastPass data not found in cache, query the database
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT expiration_date, fastpass_days
                FROM fastpass_new
                WHERE fastpass_code = %s
                """,
                (fast_pass_code,),
            )

            row = await acur.fetchone()

            if not row:
                return False

            expiration_date = row[0]
            # fastpass_cache[fast_pass_code] = expiration_date  # Store in cache

            # We need to set the code to expire at current date + fastpass_days
            if expiration_date is None:
                days_to_expire = row[1]
                background_tasks.add_task(
                    set_fastpass_expiration_date, fast_pass_code, days_to_expire
                )

            if expiration_date is None or expiration_date >= datetime.now():
                background_tasks.add_task(increment_fastpass_use_count, fast_pass_code)
                return True
            else:
                return False


class GetJobData(BaseModel):
    job_id: str


class JobRetryInfo(BaseModel):
    job_id: str


def _image_hash_insert_lock_id(job_id: Any) -> int:
    digest = hashlib.blake2b(str(job_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


async def insert_image_hashes(image_hashes, metadata, job_data):
    logging.info("Inserting image hashes")

    lora_text = ""
    for lora in metadata.get("loras") or []:
        lora_text += f"{lora.get('name')} - {lora.get('version')} - strength: {lora.get('strength')}\n"

    job_id = str(job_data.job_id)

    insert_query = """
        INSERT INTO hashes (hash, prompt, negative_prompt, seed, cfg, model, created_date, loras, job_id, finished_images_index)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = [
        (
            image_hash,
            metadata.get("prompt"),
            metadata.get("negative_prompt"),
            metadata.get("seed"),
            metadata.get("guidance_scale"),
            metadata.get("model"),
            datetime.now(),
            lora_text,
            job_id,
            index,
        )
        for index, image_hash in enumerate(image_hashes, start=1)
    ]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute("SELECT pg_advisory_xact_lock(%s)", (_image_hash_insert_lock_id(job_id),))
            await acur.execute("DELETE FROM hashes WHERE job_id = %s", (job_id,))
            await acur.executemany(insert_query, values)
            await aconn.commit()


async def twos_complement(hexstr, bits):
    value = int(hexstr, 16)  # convert hexadecimal to integer

    # convert from unsigned number to signed number with "bits" bits
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


async def process_images_and_store_hashes(image_results, metadata, job_data):
    image_hashes = []
    for image_result in image_results:
        image = decode_base64_to_image(image_result)
        image_hash = imagehash.phash(image, 8)
        image_hash = await twos_complement(str(image_hash), 64)
        image_hashes.append(image_hash)

    try:
        await insert_image_hashes(image_hashes, metadata, job_data)
    except Exception as e:
        logging.error(
            f"Error occurred while inserting image hash info into DB, JOB: {job_data.job_id}"
        )
        logging.error(str(e))


def _job_image_hash_metadata(job_details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": job_details["prompt"],
        "prompt_template": job_details.get("prompt_template"),
        "negative_prompt": job_details["negative_prompt"],
        "seed": job_details["seed"],
        "guidance_scale": job_details["guidance_scale"],
        "job_type": job_details["job_type"],
        "model": job_details["model"],
        "loras": job_details["loras"] or [],
        "lossy_images": bool(job_details["lossy_images"]),
        "regional_prompting": parse_regional_prompting(job_details["control_image"]),
    }


async def process_finished_job_images_and_store_hashes(finished_images: str, metadata: Dict[str, Any], job_data: GetJobData):
    try:
        base64_strings = [image for image in str(finished_images or "").strip("{}").split(",") if image]
        watermarked_image_base64 = []
        for base64_string in base64_strings:
            image = decode_base64_to_image(base64_string)
            watermarked_image_base64.append(
                await add_image_metadata(
                    image.convert("RGB"),
                    metadata,
                    lossy_image=metadata.get("lossy_images", False),
                )
            )

        await process_images_and_store_hashes(watermarked_image_base64, metadata, job_data)
    except Exception as e:
        logging.error(
            f"Error occurred while preparing image hash info for DB, JOB: {job_data.job_id}",
            exc_info=True,
        )


@app.post("/get_job/")
async def get_job(job_data: GetJobData, background_tasks: BackgroundTasks):
    metadata = {}
    error_message = None
    refund_info = None  # Will be populated if a refund is issued

    job_details = await get_generation_job_details(job_data.job_id)

    if not job_details:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = job_details["status"]
    queue_position = job_details["queue_position"]
    finished_images = job_details["finished_images"]
    metadata["prompt"] = job_details["prompt"]
    if job_details.get("prompt_template"):
        metadata["prompt_template"] = job_details["prompt_template"]
    metadata["negative_prompt"] = job_details["negative_prompt"]
    metadata["seed"] = job_details["seed"]
    metadata["guidance_scale"] = job_details["guidance_scale"]
    metadata["job_type"] = job_details["job_type"]
    metadata["model"] = job_details["model"]
    error_message = job_details["error_message"]
    metadata['loras'] = job_details["loras"]
    metadata['lossy_images'] = job_details["lossy_images"]
    job_user_id = job_details["user_id"]
    job_credit_cost = job_details["credit_cost"]
    job_refunded = job_details["refunded"]
    metadata['regional_prompting'] = parse_regional_prompting(job_details["control_image"])

    # Helper function to issue refund if eligible
    async def try_refund(reason: str) -> dict | None:
        """Attempt to refund credits for this job. Returns refund info or None."""
        if job_refunded or job_credit_cost <= 0 or not job_user_id:
            return None
        
        try:
            async with db_pool.connection() as refund_conn:
                async with refund_conn.cursor() as refund_cur:
                    await refund_cur.execute(
                        "SELECT * FROM safe_refund_credits(%s, %s)",
                        (job_data.job_id, reason)
                    )
                    refund_result = await refund_cur.fetchone()
                    if refund_result and refund_result[0]:  # success = True
                        logging.info(f"Refunded {refund_result[2]} credits for job {job_data.job_id}: {reason}")
                        return {
                            "credits_refunded": refund_result[2],
                            "new_balance": refund_result[1],
                            "reason": reason
                        }
        except Exception as e:
            logging.error(f"Failed to refund credits for job {job_data.job_id}: {e}")
        return None

    if job_status == "completed":

        if finished_images:
            finished_images = finished_images.strip("{}")
            base64_strings = finished_images.split(",")

            # Add watermark and metadata
            watermarked_image_base64 = []
            for i in range(4):
                image = decode_base64_to_image(base64_strings[i])
                watermarked_image_base64.append(
                    await add_image_metadata(image.convert("RGB"), metadata, lossy_image=metadata['lossy_images'])
                )

            # Generate hashes for each image and store them in DB along with image info
            # Pass the results for images and other necessary data to the background task
            background_tasks.add_task(
                process_images_and_store_hashes,
                watermarked_image_base64,
                metadata,
                job_data,
            )

            return JSONResponse(
                content={
                    "status": "completed",
                    "result": watermarked_image_base64,
                    "prompt": job_details["prompt"],
                    "prompt_template": job_details.get("prompt_template"),
                }
            )
        else:
            logging.error(
                f"Job {job_data.job_id} marked as completed but has no finished images"
            )
            # Issue refund for completed job with no images
            refund_info = await try_refund("Job completed but no images generated")
            response_content = {
                "status": "error",
                "message": "Job completed but no images found. Your credits have been refunded." if refund_info else "Job completed but no images found",
            }
            if refund_info:
                response_content["refund"] = refund_info
            return JSONResponse(content=response_content)
    elif job_status in ["pending", "processing"]:
        # Use helper_functions cache to compute ETA
        jobs_per_sec = await get_jobs_per_sec(db_pool)
        eta_seconds = compute_eta_seconds(queue_position, jobs_per_sec)
        return JSONResponse(
            content={
                "status": job_status,
                "queue_position": queue_position,
                "eta": eta_seconds,
            }
        )
    elif job_status == "failed":
        # Issue refund for failed jobs
        refund_info = await try_refund(f"Generation failed: {error_message or 'Unknown error'}")
        response_content = {
            "status": "failed", 
            "message": error_message
        }
        if refund_info:
            response_content["refund"] = refund_info
            response_content["message"] = f"{error_message or 'Generation failed'}. Your credits have been refunded."
        return JSONResponse(content=response_content)
    else:
        return JSONResponse(
            content={"status": "error", "message": "Unknown job status"}
        )


@app.post("/get_job_status/")
async def get_job_status(job_data: GetJobData, background_tasks: BackgroundTasks):
    """Lightweight status-only endpoint for polling. Returns no image data."""
    job_details = await get_generation_job_details(job_data.job_id)

    if not job_details:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = job_details["status"]
    queue_position = job_details["queue_position"]
    error_message = job_details["error_message"]
    job_user_id = job_details["user_id"]
    job_credit_cost = job_details["credit_cost"]
    job_refunded = job_details["refunded"]

    if job_status == "completed":
        if job_details["finished_images"]:
            background_tasks.add_task(
                process_finished_job_images_and_store_hashes,
                job_details["finished_images"],
                _job_image_hash_metadata(job_details),
                job_data,
            )
        return JSONResponse(content={"status": "completed"})
    elif job_status in ["pending", "processing"]:
        jobs_per_sec = await get_jobs_per_sec(db_pool)
        eta_seconds = compute_eta_seconds(queue_position, jobs_per_sec)
        return JSONResponse(
            content={
                "status": job_status,
                "queue_position": queue_position,
                "eta": eta_seconds,
            }
        )
    elif job_status == "failed":
        # Issue refund for failed jobs (same logic as /get_job/)
        refund_info = None
        if not job_refunded and job_credit_cost > 0 and job_user_id:
            try:
                async with db_pool.connection() as refund_conn:
                    async with refund_conn.cursor() as refund_cur:
                        await refund_cur.execute(
                            "SELECT * FROM safe_refund_credits(%s, %s)",
                            (job_data.job_id, f"Generation failed: {error_message or 'Unknown error'}")
                        )
                        refund_result = await refund_cur.fetchone()
                        if refund_result and refund_result[0]:
                            refund_info = {
                                "credits_refunded": refund_result[2],
                                "new_balance": refund_result[1],
                                "reason": f"Generation failed: {error_message or 'Unknown error'}"
                            }
            except Exception as e:
                logging.error(f"Failed to refund credits for job {job_data.job_id}: {e}")

        response_content: dict = {"status": "failed", "message": error_message}
        if refund_info:
            response_content["refund"] = refund_info
            response_content["message"] = f"{error_message or 'Generation failed'}. Your credits have been refunded."
        return JSONResponse(content=response_content)
    else:
        return JSONResponse(content={"status": "error", "message": "Unknown job status"})


@app.get("/get_job_image/{job_id}/{image_index}")
async def get_job_image(job_id: str, image_index: int):
    """Return a single generated image as binary. Index 0-3 for the 4 images."""
    if image_index < 0 or image_index > 3:
        raise HTTPException(status_code=400, detail="image_index must be 0-3")

    job_details = await get_generation_job_details(job_id)

    if not job_details:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = job_details["status"]
    finished_images = job_details["finished_images"]
    prompt = job_details["prompt"]
    negative_prompt = job_details["negative_prompt"]
    seed = job_details["seed"]
    guidance_scale = job_details["guidance_scale"]
    job_type = job_details["job_type"]
    model = job_details["model"]
    loras = job_details["loras"]
    lossy_images = job_details["lossy_images"]

    if job_status != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    if not finished_images:
        raise HTTPException(status_code=404, detail="No images available")

    finished_images = finished_images.strip("{}")
    base64_strings = finished_images.split(",")

    if image_index >= len(base64_strings):
        raise HTTPException(status_code=404, detail="Image index out of range")

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "job_type": job_type,
        "model": model,
        "loras": loras,
        "lossy_images": lossy_images,
        "regional_prompting": parse_regional_prompting(job_details["control_image"]),
    }

    image = decode_base64_to_image(base64_strings[image_index])
    watermarked_base64 = await add_image_metadata(image.convert("RGB"), metadata, lossy_image=lossy_images)

    # Decode the watermarked base64 back to binary
    # add_image_metadata returns "data:<mime>;base64,<data>" format
    if "," in watermarked_base64:
        header, b64_data = watermarked_base64.split(",", 1)
        media_type = "image/webp" if "webp" in header else "image/png"
    else:
        b64_data = watermarked_base64
        media_type = "image/webp" if lossy_images else "image/png"

    image_bytes = base64.b64decode(b64_data)

    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/get_loras/")
async def get_loras(status: str = "active"):
    status_norm = (status or "active").strip().lower()
    if status_norm == "inactive":
        where_clause = "WHERE (image_url IS NOT NULL OR image_blob IS NOT NULL) and is_active = false"
    elif status_norm == "all":
        where_clause = "WHERE (image_url IS NOT NULL OR image_blob IS NOT NULL)"
    else:
        # Default to active
        where_clause = "WHERE (image_url IS NOT NULL OR image_blob IS NOT NULL) and is_active = true"

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                f"""
                SELECT
                    id, name, version, base_model, download_url, is_nsfw, is_minor,
                    creator, description, version_description, tags, who_added, status,
                    trigger_words, date_added, hashes, image_url, file_path, uses,
                    is_active, last_used_date, version_id
                FROM lora_metadata
                {where_clause}
                ORDER BY uses DESC
                """
            )
            # Fetch column names
            columns = [desc[0] for desc in acur.description]
            # Fetch all rows
            rows = await acur.fetchall()
            
            # Convert to list of dictionaries
            result = [dict(zip(columns, row)) for row in rows]

    # Convert the result to a JSON-serializable format
    json_compatible_result = jsonable_encoder(result)
    
    return JSONResponse(content=json_compatible_result)

async def enhanced_filter(prompt, pattern, replacement):
    # Replace spaces with \W+ to match any non-word characters between the words
    pattern = re.sub(r" ", r"\\W+", pattern)
    return re.sub(r"(?i)\b" + pattern + r"\b", replacement, prompt)


async def promptFilter(data):
    prompt = data.prompt
    negative_prompt = data.negative_prompt

    # Common character mispellings
    corrections = {
        "cream the rabbit": [
            "creem the rabbit",
            "creme the rabbit",
            "cram the rabbit",
            "crem the rabbit",
            "craem the rabbit",
            "creamm the rabbit",
            "crema the rabbit",
            "creamie the rabbit",
        ],
        "rosy the rascal": [
            "rosey the rascal",
            "rosie the rascal",
            "rosi the rascal",
            "rosyy the rascal",
        ],
        "charmy the bee": [
            "charmi the bee",
            "charmyy the bee",
            "charmie the bee",
            "charme the bee",
        ],
        "sage": ["sagee"],
        "marine the raccoon": [
            "marin the raccoon",
            "marina the racoon",
            "marinee the raccoon",
        ],
    }

    # Update any above misspellings in the prompt with correct spelling
    for correct, misspellings in corrections.items():
        for misspelling in misspellings:
            prompt = await enhanced_filter(prompt, re.escape(misspelling), correct)

    # # If above is in prompt we grab artist list from DB and remove them if they were in the prompt
    artist_list = []
    try:
        # Connect to the database
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                # Execute the query
                await acur.execute("SELECT Artist FROM excluded_artist")
                rows = await acur.fetchall()

                # Build the artist list
                artist_list = [row[0] for row in rows]

        # Convert the entire prompt to lowercase
        lowercase_prompt = prompt.lower()

        # Check and remove any filtered phrases from the prompt
        for phrase in artist_list:
            # Convert the phrase to lowercase
            lowercase_phrase = phrase.lower()

            # Replace the lowercase phrase in the lowercase prompt
            lowercase_prompt = lowercase_prompt.replace(lowercase_phrase, "")

        # Reconstruct the original case structure of the prompt
        # by iterating over the original prompt and the lowercase prompt
        final_prompt = ""
        for orig_char, lower_char in zip(prompt, lowercase_prompt):
            if lower_char == " " and orig_char != " ":
                # If a character is replaced by space, keep the space
                final_prompt += " "
            else:
                # Otherwise, use the original character
                final_prompt += orig_char

        # Now final_prompt contains the modified prompt with original case structure
        prompt = final_prompt

    except Exception as e:
        print(f"Database error encountered: {e}")

    character_list = [
        "cream the rabbit",
        "rosy the rascal",
        "sage",
        "maria robotnik",
        "marine the raccoon",
        "charmy the bee",
    ]

    censored_tags = [
        "breast",
        "nipple",
        "pussy",
        "nsfw",
        "nudity",
        "naked",
        "loli",
        "nude",
        "ass",
        "rape",
        "sex",
        "boob",
        "sex",
        "busty",
        "tits",
        "thigh",
        "thick",
        "underwear",
        "panties",
        "upskirt",
        "cum",
        "dick",
        "topless",
        "penis",
        "blowjob",
        "ahegao",
        "nude",
        "hips",
        "areola",
        "pantyhose",
        "creampie",
        "position",
        "wet",
        "autocunnilingus",
        "squirting",
        "straddling",
        "girl on top",
        "reverse cowgirl",
        "feet",
        "toes",
        "footjob",
        "vagina",
        "clitoris",
        "furry with non-furry",
        "spread legs",
        "navel",
        "bimbo",
        "fishnet",
        "hourglass figure",
        "slut",
        "interspecies",
        "hetero",
        "tongue",
        "saliva" "anal",
        "penetration",
        "anus",
        "erection",
        "masterbation",
        "butt",
        "thighhighs",
        "lube",
        "lingerie",
        "bent over",
        "doggystyle",
        "sexy",
        "Areolae",
        "exhibitionism",
        "bottomless",
        "shirt lift",
        "no bra",
        "curvy",
        "groin",
        "clothes lift",
        "stomach",
        "spreading legs",
        "hentai",
        "penetrated",
        "masturbating",
        "masturbate",
        "horny",
        "orgasm",
        "fingering",
        "voluptuous",
        "sperm",
        "handjob",
        "gangbang",
        "ejaculation",
        "uncensored",
        "Lifting Skirt",
        "mooning",
        "hindquarters",
        "presenting",
        "porn",
        "latex",
        "fellatio",
        "oral",
        "open legs",
        "spread wide",
        "fucked",
        "fucking",
        "g-string",
        "seductive gaze",
        "dress lift",
        "cleavage",
        "provocative",
        "venus body",
        "revealing clothes",
        "oppai",
        "milf",
        "wardrobe malfunction",
        "clothing aside",
        "micro bikini",
        "thong",
        "gstring",
        "mating",
        "fuck",
        "tentacle",
        "moan",
        "facial",
        "swimsuit to the side",
        "ripped dress",
        "giant chest",
        "Titjob",
        "lesbian",
        "french kiss",
        "furry with furry",
        "clit",
        "Vulva",
        "lust",
        "Libido",
        "Garter",
        "striptease",
        "cock",
        "plump",
        "thicc",
        "scissoring",
        "skimpy",
        "Anal",
        "curvaceous",
        "gaping",
        "string bikini",
        "cunnilingus",
        "Panty",
        "cameltoe",
        "dominatrix",
        "Corset",
        "lewd",
        "Explicit",
        "futanari",
        "foreskin",
        "urethra",
        "skirt lift",
        "bedroom eyes",
        "pregnant",
        "nudist",
        "undressing",
        "black bra",
        "aroused",
        "yuri",
        "d-cup",
        "skindentation",
        "seductive",
        "booty",
        "big melons",
        "testicles",
        "bodily fluid",
        "semen",
        "erect",
        "twerking",
        "lactating",
        "stockings",
        "cowgirl posicion",
        "vaginia",
        "masturbation",
        "pants pull",
        "clothes pull",
        "genital",
        "ming",
        "nudly",
        "breeding",
        "orgy",
        "pinned down",
        "thrusting",
        "cervical",
        "ecstasy",
    ]

    # If character is in prompt, filter out censored tags from prompt
    if any(character in prompt.lower() for character in character_list):
        for tag in censored_tags:
            prompt = prompt.lower().replace(tag.lower(), "")

        # If prompt is changed remove the prompt "blush" from prompt
        if prompt != data.prompt.lower():
            prompt = prompt.replace("blush", "")

        negative_prompt = (
            "(cleavage), navel, 3d, blush, sweat, ((underwear)), (bikini), (nipples), sex, (breasts), nude, "
            + negative_prompt
        )
        logging.error(prompt)

    return prompt, negative_prompt


async def fortify_default_negative(negative_prompt):
    if "nsfw" in negative_prompt.lower() and "nipples" not in negative_prompt.lower():
        return "nipples, pussy, breasts, " + negative_prompt
    else:
        return negative_prompt


@app.post("/rate_image/")
async def rate_image(job_data: JobData):
    return JSONResponse({"message": f"we no longer use this"})


class Subscription(BaseModel):
    userId: str
    endpoint: str
    expirationTime: Optional[str]
    keys: dict


def _vapid_claims() -> dict:
    # pywebpush requires a `sub` (mailto: or https:) claim.
    sub = VAPID_CLAIMS or "mailto:admin@mobians.ai"
    if not sub.startswith(("mailto:", "https:")):
        sub = f"mailto:{sub}"
    return {"sub": sub}


async def _delete_push_subscription(endpoint: str) -> None:
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    "DELETE FROM push_subscriptions WHERE endpoint = %s",
                    (endpoint,),
                )
                await aconn.commit()
    except Exception as exc:
        logging.warning(f"Failed to prune dead push subscription {endpoint}: {exc}")


async def _send_webpush(endpoint: str, p256dh: str, auth: str, payload: dict) -> bool:
    """Send a single web push. Returns True on success. Prunes dead subs."""
    if not VAPID_PRIVATE_KEY:
        logging.warning("VAPID_PRIVATE_KEY not configured; skipping push notification")
        return False
    try:
        webpush(
            subscription_info={
                "endpoint": endpoint,
                "keys": {"p256dh": p256dh, "auth": auth},
            },
            data=json.dumps(payload),
            ttl=WEBPUSH_TTL_SECONDS,
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims=_vapid_claims(),
        )
        return True
    except WebPushException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (404, 410):
            # Endpoint gone permanently — drop it.
            await _delete_push_subscription(endpoint)
        logging.warning(f"webpush failed (status={status}) for {endpoint}: {exc}")
        return False
    except Exception as exc:
        logging.error(f"webpush unexpected error for {endpoint}: {exc}")
        return False


async def send_push_to_user(user_id: str, payload: dict) -> int:
    """Send the given payload to every subscription owned by user_id."""
    if not user_id:
        return 0
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                "SELECT endpoint, p256dh, auth FROM push_subscriptions WHERE user_id = %s",
                (user_id,),
            )
            rows = await acur.fetchall()
    sent = 0
    for endpoint, p256dh, auth in rows:
        if await _send_webpush(endpoint, p256dh, auth, payload):
            sent += 1
    return sent


async def send_push_to_anonymous(anonymous_id: str, payload: dict) -> int:
    """Send the given payload to anonymous (not-logged-in) subscriptions."""
    if not anonymous_id:
        return 0
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT endpoint, p256dh, auth
                FROM push_subscriptions
                WHERE anonymous_id = %s AND user_id IS NULL
                """,
                (anonymous_id,),
            )
            rows = await acur.fetchall()
    sent = 0
    for endpoint, p256dh, auth in rows:
        if await _send_webpush(endpoint, p256dh, auth, payload):
            sent += 1
    return sent


@app.post("/subscribe")
async def subscribe(
    subscription: Subscription,
    user: Optional[dict] = Depends(get_current_user),
):
    """Register (or refresh) a Web Push subscription.

    If the caller is authenticated, the subscription is linked to their user_id
    so we can target them across devices (e.g., LoRA-approved notifications).
    Anonymous browsers are keyed by the client-generated `userId`.
    """
    p256dh = subscription.keys.get("p256dh") if subscription.keys else None
    auth = subscription.keys.get("auth") if subscription.keys else None
    if not (subscription.endpoint and p256dh and auth):
        raise HTTPException(status_code=400, detail="Invalid subscription payload")

    user_id = user["user_id"] if user else None
    anonymous_id = subscription.userId or None

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                INSERT INTO push_subscriptions (user_id, anonymous_id, endpoint, p256dh, auth)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (endpoint) DO UPDATE SET
                    user_id      = COALESCE(EXCLUDED.user_id, push_subscriptions.user_id),
                    anonymous_id = COALESCE(EXCLUDED.anonymous_id, push_subscriptions.anonymous_id),
                    p256dh       = EXCLUDED.p256dh,
                    auth         = EXCLUDED.auth,
                    last_used_at = NOW()
                """,
                (user_id, anonymous_id, subscription.endpoint, p256dh, auth),
            )
            await aconn.commit()

    return {"status": "subscribed", "linked_to_user": bool(user_id)}


@app.post("/unsubscribe")
async def unsubscribe(payload: Dict[str, Any]):
    """Remove a push subscription by endpoint."""
    endpoint = (payload or {}).get("endpoint")
    if not endpoint:
        raise HTTPException(status_code=400, detail="endpoint is required")
    await _delete_push_subscription(endpoint)
    return {"status": "unsubscribed"}


@app.get("/send_notification/{user_id}")
async def send_notification(user_id: str):
    """Send the 'image ready' notification.

    `user_id` here is whatever id the caller stored at subscribe-time. It may
    be a real authenticated user UUID OR the client-side anonymous id.
    We try both so the existing frontend (anon trigger on poll-complete) and
    server-initiated flows both work.
    """
    payload = {
        "notification": {
            "title": "Your image is ready!",
            "body": "Click to view your generation.",
            "vibrate": [100, 50, 100],
            "data": {"url": PUBLIC_SITE_URL},
        }
    }

    sent = 0
    # Authenticated match (user_id is a UUID string).
    try:
        sent += await send_push_to_user(user_id, payload)
    except Exception as exc:
        logging.warning(f"send_push_to_user failed: {exc}")
    # Anonymous fallback (legacy behavior).
    sent += await send_push_to_anonymous(user_id, payload)

    if sent == 0:
        return {"status": "failed", "detail": "No active subscriptions"}
    return {"status": "sent", "count": sent}


# ============================================
# INTERNAL NOTIFICATION ENDPOINTS
# ============================================

def _require_internal_token(x_internal_token: Optional[str]) -> None:
    if not INTERNAL_API_TOKEN:
        raise HTTPException(status_code=503, detail="Internal notifications disabled")
    if not x_internal_token or x_internal_token != INTERNAL_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid internal token")


async def _resolve_user_id_from_requestor(requestor: Optional[str]) -> Optional[str]:
    """Map a lora_suggestions.requestor value back to a users.id.

    `requestor` can be a UUID (user_id), discord id, google id, username, or email.
    """
    if not requestor:
        return None
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT id FROM users
                WHERE id::text         = %s
                   OR discord_user_id  = %s
                   OR google_user_id   = %s
                   OR username         = %s
                   OR email            = %s
                LIMIT 1
                """,
                (requestor, requestor, requestor, requestor, requestor),
            )
            row = await acur.fetchone()
    return str(row[0]) if row else None


class LoraNotifyPayload(BaseModel):
    version_id: int
    name: Optional[str] = None
    version: Optional[str] = None
    requestor: Optional[str] = None


@app.post("/internal/notify_lora_downloaded")
async def internal_notify_lora_downloaded(
    payload: LoraNotifyPayload,
    x_internal_token: Optional[str] = Header(default=None, alias="X-Internal-Token"),
):
    """Called by the LoRA downloader service once a LoRA is available on-site."""
    _require_internal_token(x_internal_token)

    requestor = payload.requestor
    # Fallback: look up requestor from the suggestion if not supplied.
    if not requestor:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    "SELECT requestor, name, version FROM lora_suggestions WHERE version_id = %s",
                    (payload.version_id,),
                )
                row = await acur.fetchone()
        if row:
            requestor = row[0]
            payload.name = payload.name or row[1]
            payload.version = payload.version or row[2]

    user_id = await _resolve_user_id_from_requestor(requestor)
    if not user_id:
        return {"status": "skipped", "reason": "no user_id for requestor"}

    lora_label = payload.name or "Your LoRA"
    if payload.version:
        lora_label = f"{lora_label} v{payload.version}"

    notification_payload = {
        "notification": {
            "title": "Your LoRA is ready!",
            "body": f"{lora_label} is now available on Mobians.ai.",
            "vibrate": [100, 50, 100],
            "data": {"url": PUBLIC_SITE_URL},
        }
    }
    sent = await send_push_to_user(user_id, notification_payload)
    return {"status": "sent" if sent else "no_subscriptions", "count": sent}


class DiscordAuthCode(BaseModel):
    code: str
    redirect_uri: Optional[str] = None
    link: Optional[bool] = False  # Allow link parameter from frontend


@app.post("/discord_auth/")
async def discord_auth(auth_code: DiscordAuthCode):
    discord_token_url = "https://discord.com/api/oauth2/token"
    client_id = os.environ.get("DISCORD_CLIENT_ID")
    client_secret = os.environ.get("DISCORD_CLIENT_SECRET")
    redirect_uri = auth_code.redirect_uri or os.environ.get("DISCORD_REDIRECT_URI")

    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(
            status_code=500,
            detail="Discord OAuth not fully configured on server",
        )

    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": auth_code.code,
        "redirect_uri": redirect_uri,
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    # Use a fresh session without trust_env to avoid proxy issues
    try:
        async with aiohttp.ClientSession() as oauth_session:
            async with oauth_session.post(discord_token_url, data=data, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logging.error(f"Discord token exchange failed: {resp.status} - {error_text}")
                    raise HTTPException(
                        status_code=resp.status, detail=f"Error in Discord token exchange: {error_text}"
                    )
                token_data = await resp.json()
                access_token = token_data.get("access_token")

            discord_guilds_url = "https://discord.com/api/users/@me/guilds"
            auth_headers = {"Authorization": f"Bearer {access_token}"}

            async with oauth_session.get(discord_guilds_url, headers=auth_headers) as guild_resp:
                if guild_resp.status != 200:
                    raise HTTPException(
                        status_code=guild_resp.status,
                        detail="Error fetching user guilds from Discord",
                    )
                guilds = await guild_resp.json()

            # Fetch the authenticated user's information
            discord_user_url = "https://discord.com/api/users/@me"
            async with oauth_session.get(discord_user_url, headers=auth_headers) as user_resp:
                if user_resp.status != 200:
                    raise HTTPException(
                        status_code=user_resp.status,
                        detail="Error fetching user data from Discord",
                    )
                user_data = await user_resp.json()
    except aiohttp.ClientError as e:
        logging.error(f"Discord API connection error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to Discord API: {str(e)}"
        )

    user_id = user_data.get("id")  # Get the user's ID
    username = user_data.get("username")
    global_name = user_data.get("global_name")

    your_guild_id = "1095514548112461924"  # Replace with your Discord server's ID
    is_member_of_your_guild = any(guild["id"] == your_guild_id for guild in guilds)
    role_ids_to_check = [
        "1100272052008652922",
    ]

    has_required_role = False
    bot_ip = os.environ.get("DISCORD_BOT_IP")
    # Optionally query bot for role check; don't fail login if bot is unavailable
    if bot_ip and user_id:
        try:
            async with session.post(
                f"http://{bot_ip}:6965/check_role",
                json={
                    "guild_id": your_guild_id,
                    "user_id": user_id,
                    "role_ids": role_ids_to_check,
                },
                headers={"Authorization": "YourSecretToken"},
            ) as bot_resp:
                if bot_resp.status == 200:
                    bot_data = await bot_resp.json()
                    has_required_role = bool(bot_data.get("has_role"))
        except Exception as e:
            logging.error(f"Bot role check failed: {e}")

    # Build Discord avatar URL
    avatar_hash = user_data.get("avatar")
    avatar_url = None
    if avatar_hash:
        avatar_url = f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png"

    # Create or update user in database and get JWT token
    try:
        user_record = await upsert_user(
            discord_user_id=user_id,
            username=username,
            display_name=global_name or username,
            avatar_url=avatar_url
        )
    except Exception as e:
        logging.error(f"Error upserting user: {e}")
        # Fall back to returning basic info without credits
        return {
            "status": "success",
            "is_member_of_your_guild": is_member_of_your_guild,
            "has_required_role": has_required_role,
            "discord_user_id": user_id,
            "username": username,
            "display_name": global_name or username,
        }

    return {
        "status": "success",
        "is_member_of_your_guild": is_member_of_your_guild,
        "has_required_role": has_required_role,
        "discord_user_id": user_id,
        "username": username,
        "display_name": global_name or username,
        "avatar_url": avatar_url,
        # New fields for credits system
        "user_id": user_record["user_id"],
        "credits": user_record["credits"],
        "token": user_record["token"],
        "last_daily_bonus": user_record["last_daily_bonus"],
        "daily_bonus_streak": user_record["daily_bonus_streak"],
        "is_new_user": user_record["is_new_user"],
    }


class GoogleAuthCode(BaseModel):
    code: str
    redirect_uri: Optional[str] = None
    link: Optional[bool] = False  # Allow link parameter from frontend


@app.post("/google_auth/")
async def google_auth(auth_code: GoogleAuthCode):
    token_url = "https://oauth2.googleapis.com/token"
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = auth_code.redirect_uri or os.environ.get("GOOGLE_REDIRECT_URI")

    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(status_code=500, detail="Google OAuth not configured on server")

    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": auth_code.code,
        "redirect_uri": redirect_uri,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        async with aiohttp.ClientSession() as oauth_session:
            async with oauth_session.post(token_url, data=data, headers=headers) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    logging.error(f"Google token exchange failed: {resp.status} - {detail}")
                    raise HTTPException(status_code=resp.status, detail=f"Error in Google token exchange: {detail}")
                token_data = await resp.json()

            access_token = token_data.get("access_token")
            if not access_token:
                raise HTTPException(status_code=400, detail="Missing access_token in Google response")

            userinfo_url = "https://openidconnect.googleapis.com/v1/userinfo"
            auth_headers = {"Authorization": f"Bearer {access_token}"}
            async with oauth_session.get(userinfo_url, headers=auth_headers) as user_resp:
                if user_resp.status != 200:
                    raise HTTPException(status_code=user_resp.status, detail="Error fetching Google userinfo")
                user_data = await user_resp.json()
    except aiohttp.ClientError as e:
        logging.error(f"Google API connection error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to Google API: {str(e)}"
        )

    # Create or update user in database and get JWT token
    try:
        user_record = await upsert_user(
            google_user_id=user_data.get("sub"),
            email=user_data.get("email"),
            username=user_data.get("email"),
            display_name=user_data.get("name") or user_data.get("email"),
            avatar_url=user_data.get("picture")
        )
    except Exception as e:
        logging.error(f"Error upserting user: {e}")
        # Fall back to returning basic info without credits
        return {
            "status": "success",
            "google_user_id": user_data.get("sub"),
            "email": user_data.get("email"),
            "username": user_data.get("email"),
            "display_name": user_data.get("name") or user_data.get("email"),
            "picture": user_data.get("picture"),
        }

    # user_data contains fields like: sub, email, email_verified, name, given_name, family_name, picture
    return {
        "status": "success",
        "google_user_id": user_data.get("sub"),
        "email": user_data.get("email"),
        "username": user_data.get("email"),
        "display_name": user_data.get("name") or user_data.get("email"),
        "picture": user_data.get("picture"),
        # New fields for credits system
        "user_id": user_record["user_id"],
        "credits": user_record["credits"],
        "token": user_record["token"],
        "last_daily_bonus": user_record["last_daily_bonus"],
        "daily_bonus_streak": user_record["daily_bonus_streak"],
        "is_new_user": user_record["is_new_user"],
    }


# ============================================
# CREDIT MANAGEMENT ENDPOINTS
# ============================================

@app.get("/user/me")
async def get_current_user_info(user: dict = Depends(require_auth)):
    """Get the current authenticated user's information including credits."""
    return {
        "status": "success",
        "user": user
    }


@app.get("/user/credits")
async def get_user_credits(user: dict = Depends(require_auth)):
    """Get the current user's credit balance and recent transactions."""
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Get recent transactions
            await acur.execute(
                """
                SELECT id, amount, balance_after, transaction_type, description, created_at
                FROM credit_transactions
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (user_id,)
            )
            rows = await acur.fetchall()
            
            transactions = [
                {
                    "id": str(row[0]),
                    "amount": row[1],
                    "balance_after": row[2],
                    "type": row[3],
                    "description": row[4],
                    "created_at": row[5].isoformat() if row[5] else None
                }
                for row in rows
            ]
    
    # Check daily bonus state using the DB's current_date to avoid timezone drift
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT
                    (last_daily_bonus IS NULL OR last_daily_bonus <> CURRENT_DATE) AS can_claim_daily,
                    last_daily_bonus,
                    daily_bonus_streak,
                    CASE
                        WHEN last_daily_bonus = CURRENT_DATE OR last_daily_bonus = CURRENT_DATE - 1
                            THEN LEAST(30 + daily_bonus_streak * 10, 50)
                        ELSE 30
                    END AS next_daily_bonus,
                    CASE
                        WHEN last_daily_bonus = CURRENT_DATE OR last_daily_bonus = CURRENT_DATE - 1
                            THEN daily_bonus_streak + 1
                        ELSE 1
                    END AS next_daily_streak
                FROM users
                WHERE id = %s
                """,
                (user_id,)
            )
            row = await acur.fetchone()
            can_claim_daily = bool(row[0]) if row else False
            last_daily_bonus = row[1].isoformat() if row and row[1] else None
            daily_bonus_streak = int(row[2]) if row and row[2] is not None else user.get("daily_bonus_streak", 0)
            next_daily_bonus = int(row[3]) if row and row[3] is not None else 30
            next_daily_streak = int(row[4]) if row and row[4] is not None else 1

    return {
        "status": "success",
        "credits": user["credits"],
        "can_claim_daily_bonus": can_claim_daily,
        "daily_bonus_streak": daily_bonus_streak,
        "last_daily_bonus": last_daily_bonus,
        "next_daily_bonus": next_daily_bonus,
        "next_daily_bonus_streak": next_daily_streak,
        "transactions": transactions
    }


@app.post("/user/credits/daily")
async def claim_daily_bonus(user: dict = Depends(require_auth)):
    """Claim the daily credit bonus."""
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Call the stored function
            await acur.execute(
                "SELECT * FROM claim_daily_bonus(%s)",
                (user_id,)
            )
            result = await acur.fetchone()
            
            if result:
                success, credits_awarded, new_balance, streak, message = result
                
                if success:
                    return {
                        "status": "success",
                        "message": message,
                        "credits_awarded": credits_awarded,
                        "new_balance": new_balance,
                        "streak": streak
                    }
                else:
                    raise HTTPException(status_code=400, detail=message)
            else:
                raise HTTPException(status_code=500, detail="Failed to claim daily bonus")


@app.get("/credits/cost/{model}")
async def get_model_credit_cost(model: str):
    """Get the credit cost for a specific model. Public endpoint."""
    cost = get_credit_cost(model)
    base_type = MODEL_BASE_TYPES.get(model, "SD 1.5")
    per_lora_cost = LORA_CREDIT_COSTS.get(base_type, 0)
    
    return {
        "model": model,
        "base_type": base_type,
        "credit_cost": cost,
        "per_lora_credit_cost": per_lora_cost,
    }


@app.get("/credits/costs")
async def get_all_credit_costs():
    """Get credit costs for all models. Public endpoint."""
    return {
        "costs": CREDIT_COSTS,
        "lora_costs": LORA_CREDIT_COSTS,
        "models": MODEL_BASE_TYPES
    }


# ============================================
# PAYPAL PAYMENT ENDPOINTS
# ============================================

async def get_paypal_access_token() -> str:
    """Get PayPal OAuth access token."""
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise HTTPException(status_code=503, detail="PayPal not configured")
    
    auth = aiohttp.BasicAuth(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = "grant_type=client_credentials"
    
    async with aiohttp.ClientSession() as pp_session:
        async with pp_session.post(
            f"{PAYPAL_API_BASE}/v1/oauth2/token",
            auth=auth,
            headers=headers,
            data=data
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logging.error(f"PayPal auth failed: {error_text}")
                raise HTTPException(status_code=503, detail="PayPal authentication failed")
            token_data = await resp.json()
            return token_data["access_token"]


@app.get("/credit-packages")
async def get_credit_packages():
    """Get available credit packages for purchase. Public endpoint."""
    return {
        "packages": list(CREDIT_PACKAGES.values()),
        "paypal_client_id": PAYPAL_CLIENT_ID,
        "paypal_mode": PAYPAL_MODE
    }


class CreateOrderRequest(BaseModel):
    package_id: str


@app.post("/paypal/create-order")
async def paypal_create_order(request: CreateOrderRequest, user: dict = Depends(require_auth)):
    """Create a PayPal order for a credit package."""
    package = CREDIT_PACKAGES.get(request.package_id)
    if not package:
        raise HTTPException(status_code=400, detail="Invalid package ID")
    
    access_token = await get_paypal_access_token()
    
    order_data = {
        "intent": "CAPTURE",
        "purchase_units": [{
            "reference_id": f"{user['user_id']}_{request.package_id}_{int(time.time())}",
            "description": package["description"],
            "amount": {
                "currency_code": "USD",
                "value": f"{package['price_usd']:.2f}"
            },
            "custom_id": json.dumps({
                "user_id": user["user_id"],
                "package_id": request.package_id,
                "credits": package["credits"]
            })
        }]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    async with aiohttp.ClientSession() as pp_session:
        async with pp_session.post(
            f"{PAYPAL_API_BASE}/v2/checkout/orders",
            headers=headers,
            json=order_data
        ) as resp:
            if resp.status not in (200, 201):
                error_text = await resp.text()
                logging.error(f"PayPal create order failed: {error_text}")
                raise HTTPException(status_code=500, detail="Failed to create PayPal order")
            order = await resp.json()
            
            return {
                "order_id": order["id"],
                "status": order["status"]
            }


class CaptureOrderRequest(BaseModel):
    order_id: str


@app.post("/paypal/capture-order")
async def paypal_capture_order(request: CaptureOrderRequest, user: dict = Depends(require_auth)):
    """Capture a PayPal order after user approval and credit the user's account."""
    access_token = await get_paypal_access_token()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    async with aiohttp.ClientSession() as pp_session:
        # Capture the order
        async with pp_session.post(
            f"{PAYPAL_API_BASE}/v2/checkout/orders/{request.order_id}/capture",
            headers=headers
        ) as resp:
            if resp.status not in (200, 201):
                error_text = await resp.text()
                logging.error(f"PayPal capture failed: {error_text}")
                raise HTTPException(status_code=500, detail="Failed to capture PayPal order")
            
            capture_data = await resp.json()
            
            if capture_data["status"] != "COMPLETED":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Order not completed. Status: {capture_data['status']}"
                )
            
            # Extract custom data from the purchase unit
            purchase_unit = capture_data["purchase_units"][0]
            custom_data = json.loads(purchase_unit.get("payments", {}).get("captures", [{}])[0].get("custom_id", "{}"))
            
            # If custom_id is not in captures, try the purchase unit level
            if not custom_data:
                custom_data = json.loads(purchase_unit.get("custom_id", "{}"))
            
            # Verify the user matches
            if custom_data.get("user_id") != user["user_id"]:
                logging.error(f"User mismatch: order user {custom_data.get('user_id')} != auth user {user['user_id']}")
                raise HTTPException(status_code=403, detail="Order does not belong to this user")
            
            package_id = custom_data.get("package_id")
            credits_to_add = custom_data.get("credits", 0)
            
            if not credits_to_add:
                package = CREDIT_PACKAGES.get(package_id)
                if package:
                    credits_to_add = package["credits"]
            
            # Get PayPal transaction ID for reference
            paypal_capture_id = purchase_unit.get("payments", {}).get("captures", [{}])[0].get("id", request.order_id)
            
            # Add credits to user's account
            async with db_pool.connection() as aconn:
                async with aconn.cursor() as acur:
                    # Check if this order was already processed (idempotency)
                    await acur.execute(
                        "SELECT id FROM credit_transactions WHERE payment_reference = %s",
                        (paypal_capture_id,)
                    )
                    existing = await acur.fetchone()
                    if existing:
                        # Already processed, return success without double-crediting
                        await acur.execute(
                            "SELECT credits FROM users WHERE id = %s",
                            (user["user_id"],)
                        )
                        current_credits = (await acur.fetchone())[0]
                        return {
                            "status": "success",
                            "message": "Order already processed",
                            "credits_added": credits_to_add,
                            "new_balance": current_credits
                        }
                    
                    # Add credits
                    await acur.execute(
                        """
                        UPDATE users SET credits = credits + %s WHERE id = %s
                        RETURNING credits
                        """,
                        (credits_to_add, user["user_id"])
                    )
                    new_balance = (await acur.fetchone())[0]
                    
                    # Record transaction
                    await acur.execute(
                        """
                        INSERT INTO credit_transactions 
                        (user_id, amount, balance_after, transaction_type, description, payment_provider, payment_reference)
                        VALUES (%s, %s, %s, 'purchase', %s, 'paypal', %s)
                        """,
                        (
                            user["user_id"],
                            credits_to_add,
                            new_balance,
                            f"Purchased {CREDIT_PACKAGES.get(package_id, {}).get('name', 'Credit Pack')}",
                            paypal_capture_id
                        )
                    )
                    
                    await aconn.commit()
            
            return {
                "status": "success",
                "message": f"Successfully added {credits_to_add} credits!",
                "credits_added": credits_to_add,
                "new_balance": new_balance,
                "package": CREDIT_PACKAGES.get(package_id, {}).get("name")
            }


# ============================================
# ADMIN ENDPOINTS
# ============================================

# Discord role IDs for admin/mod access
ADMIN_ROLE_IDS = [
    "1100272052008652922",
]


async def require_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Dependency that requires admin/mod privileges.
    Checks Discord role via bot or database flag.
    """
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if user.get("is_banned"):
        raise HTTPException(status_code=403, detail="Account is banned")
    
    # Check if user has admin role - first check database, then Discord bot
    discord_user_id = user.get("discord_user_id")
    if not discord_user_id:
        raise HTTPException(status_code=403, detail="Admin access requires Discord login")
    
    # Query bot for role check
    bot_ip = os.environ.get("DISCORD_BOT_IP")
    your_guild_id = "1095514548112461924"
    
    if not bot_ip:
        raise HTTPException(status_code=503, detail="Admin verification unavailable")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{bot_ip}:6965/check_role",
                json={
                    "guild_id": your_guild_id,
                    "user_id": discord_user_id,
                    "role_ids": ADMIN_ROLE_IDS,
                },
                headers={"Authorization": "YourSecretToken"},
            ) as bot_resp:
                if bot_resp.status == 200:
                    bot_data = await bot_resp.json()
                    if bot_data.get("has_role"):
                        return user
    except Exception as e:
        logging.error(f"Bot role check failed: {e}")
        raise HTTPException(status_code=503, detail="Admin verification failed")
    
    raise HTTPException(status_code=403, detail="Admin access denied")


@app.get("/admin/dynamic-prompts/library")
async def admin_get_dynamic_prompt_library(user: dict = Depends(require_admin)):
    assets = await _load_dynamic_prompt_assets(include_inactive=True)
    return JSONResponse(content=assets["admin_library"])


@app.put("/admin/dynamic-prompts/library")
async def admin_update_dynamic_prompt_library(
    request: DynamicPromptAdminLibraryUpdate,
    user: dict = Depends(require_admin),
):
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            for index, category in enumerate(request.categories):
                category_id = category.id.strip()
                if not category_id:
                    continue
                token = category.token or f"_{category_id}_"
                display_order = category.display_order if category.display_order is not None else (index + 1) * 10
                await acur.execute(
                    """
                    INSERT INTO dynamic_prompt.categories (
                        id, wildcard_set, label, token, description, display_order, is_active, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        wildcard_set = EXCLUDED.wildcard_set,
                        label = EXCLUDED.label,
                        token = EXCLUDED.token,
                        description = EXCLUDED.description,
                        display_order = EXCLUDED.display_order,
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW()
                    """,
                    (
                        category_id,
                        WILDCARD_SET_ID,
                        category.label.strip() or category_id,
                        token,
                        category.description or "",
                        display_order,
                        bool(category.is_active),
                    ),
                )

                await acur.execute(
                    "UPDATE dynamic_prompt.items SET is_active = FALSE, updated_at = NOW() WHERE category_id = %s",
                    (category_id,),
                )

                seen_entries: set[str] = set()
                for entry_index, raw_entry in enumerate(category.entries or []):
                    entry = str(raw_entry).strip()
                    if not entry or entry in seen_entries:
                        continue
                    seen_entries.add(entry)
                    await acur.execute(
                        """
                        INSERT INTO dynamic_prompt.items (category_id, value, display_order, is_active, updated_at)
                        VALUES (%s, %s, %s, TRUE, NOW())
                        ON CONFLICT (category_id, value) DO UPDATE SET
                            display_order = EXCLUDED.display_order,
                            is_active = TRUE,
                            updated_at = NOW()
                        """,
                        (category_id, entry, (entry_index + 1) * 10),
                    )

            for index, starter in enumerate(request.starter_templates):
                starter_id = starter.id.strip()
                if not starter_id or not starter.template.strip():
                    continue
                token = starter.token or _starter_template_token(starter_id)
                display_order = starter.display_order if starter.display_order is not None else (index + 1) * 10
                await acur.execute(
                    """
                    INSERT INTO dynamic_prompt.starter_templates (
                        id, wildcard_set, name, description, token, template, display_order, is_active, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        wildcard_set = EXCLUDED.wildcard_set,
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        token = EXCLUDED.token,
                        template = EXCLUDED.template,
                        display_order = EXCLUDED.display_order,
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW()
                    """,
                    (
                        starter_id,
                        WILDCARD_SET_ID,
                        starter.name.strip() or starter_id,
                        starter.description or "",
                        token,
                        starter.template.strip(),
                        display_order,
                        bool(starter.is_active),
                    ),
                )

        await aconn.commit()

    _invalidate_dynamic_prompt_cache()
    assets = await _load_dynamic_prompt_assets(include_inactive=True)
    return JSONResponse(content=assets["admin_library"])


@app.get("/admin/dynamic-prompts/templates")
async def admin_list_dynamic_prompt_templates(
    status: str = "approved",
    user: dict = Depends(require_admin),
):
    normalized_status = status.strip().lower()
    if normalized_status != "all" and normalized_status not in ADMIN_COMMUNITY_TEMPLATE_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid template status filter.")

    params: List[Any] = []
    where_clause = " WHERE t.status <> 'private'"
    if normalized_status != "all":
        where_clause = " WHERE t.status = %s"
        params.append(normalized_status)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _community_template_select_sql(None)
                + where_clause
                + " ORDER BY t.submitted_at DESC NULLS LAST, t.updated_at DESC LIMIT 200",
                tuple(params),
            )
            rows = await acur.fetchall()

    templates = []
    for row in rows:
        item = _community_template_response(row)
        try:
            item["preview_samples"] = await _community_template_preview_samples(item["template"], item["user_id"])
        except DynamicPromptError as exc:
            item["preview_error"] = str(exc)
        templates.append(item)

    return JSONResponse(content={"templates": templates, "status": normalized_status})


@app.get("/admin/dynamic-prompts/categories")
async def admin_list_dynamic_prompt_categories(
    status: str = "public",
    user: dict = Depends(require_admin),
):
    normalized_status = status.strip().lower()
    if normalized_status != "all" and normalized_status not in ADMIN_CUSTOM_CATEGORY_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid category status filter.")

    params: List[Any] = []
    where_clause = " WHERE c.status <> 'private'"
    if normalized_status != "all":
        where_clause = " WHERE c.status = %s"
        params.append(normalized_status)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                _custom_category_select_sql(None)
                + where_clause
                + " ORDER BY c.updated_at DESC LIMIT 200",
                tuple(params),
            )
            rows = await acur.fetchall()

    return JSONResponse(
        content={
            "categories": [_custom_category_response(row) for row in rows],
            "status": normalized_status,
        }
    )


@app.post("/admin/dynamic-prompts/templates/{template_id}/approve")
async def admin_approve_dynamic_prompt_template(template_id: str, user: dict = Depends(require_admin)):
    existing = await _fetch_community_template(template_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found.")
    if existing["status"] != "pending":
        raise HTTPException(status_code=409, detail="Only pending templates can be approved.")

    preview_samples = (await _validate_community_dynamic_prompt_template(
        existing["title"],
        existing["description"],
        existing["template"],
        existing["tags"],
        user_id=existing["user_id"],
    ))["preview_samples"]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.templates
                SET status = 'approved', rejection_reason = NULL, approved_at = NOW(), hidden_at = NULL, updated_at = NOW()
                WHERE id = %s AND status = 'pending'
                """,
                (template_id,),
            )
            await aconn.commit()

    approved = await _fetch_community_template(template_id)
    return JSONResponse(content={"template": approved, "preview_samples": preview_samples})


@app.post("/admin/dynamic-prompts/templates/{template_id}/reject")
async def admin_reject_dynamic_prompt_template(
    template_id: str,
    request: DynamicPromptTemplateRejectRequest,
    user: dict = Depends(require_admin),
):
    reason = request.reason.strip()
    if len(reason) < 3:
        raise HTTPException(status_code=400, detail="A rejection reason is required.")
    if len(reason) > COMMUNITY_TEMPLATE_DESCRIPTION_MAX:
        raise HTTPException(status_code=400, detail=f"Rejection reason must be {COMMUNITY_TEMPLATE_DESCRIPTION_MAX} characters or fewer.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.templates
                SET status = 'rejected', rejection_reason = %s, approved_at = NULL, updated_at = NOW()
                WHERE id = %s AND status = 'pending'
                RETURNING id
                """,
                (reason, template_id),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Pending template not found.")
            await aconn.commit()

    rejected = await _fetch_community_template(template_id)
    return JSONResponse(content={"template": rejected})


@app.post("/admin/dynamic-prompts/templates/{template_id}/hide")
async def admin_hide_dynamic_prompt_template(template_id: str, user: dict = Depends(require_admin)):
    existing = await _fetch_community_template(template_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found.")
    if existing["status"] == "private":
        raise HTTPException(status_code=409, detail="Private templates cannot be moderated.")
    if existing["status"] == "hidden":
        raise HTTPException(status_code=409, detail="Template is already hidden.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.templates
                SET status = 'hidden', hidden_at = NOW(), updated_at = NOW()
                WHERE id = %s AND status IN ('pending', 'approved', 'rejected')
                RETURNING id
                """,
                (template_id,),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Template not found.")
            await aconn.commit()

    hidden = await _fetch_community_template(template_id)
    return JSONResponse(content={"template": hidden})


@app.post("/admin/dynamic-prompts/templates/{template_id}/restore")
async def admin_restore_dynamic_prompt_template(template_id: str, user: dict = Depends(require_admin)):
    existing = await _fetch_community_template(template_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found.")
    if existing["status"] != "hidden":
        raise HTTPException(status_code=409, detail="Only hidden templates can be restored.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.templates
                SET status = CASE
                        WHEN approved_at IS NOT NULL THEN 'approved'
                        WHEN rejection_reason IS NOT NULL THEN 'rejected'
                        ELSE 'pending'
                    END,
                    hidden_at = NULL,
                    updated_at = NOW()
                WHERE id = %s AND status = 'hidden'
                RETURNING id
                """,
                (template_id,),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Hidden template not found.")
            await aconn.commit()

    restored = await _fetch_community_template(template_id)
    return JSONResponse(content={"template": restored})


@app.post("/admin/dynamic-prompts/categories/{category_id}/hide")
async def admin_hide_dynamic_prompt_category(category_id: str, user: dict = Depends(require_admin)):
    existing = await _fetch_custom_category(category_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found.")
    if existing["status"] == "private":
        raise HTTPException(status_code=409, detail="Private categories cannot be moderated.")
    if existing["status"] == "hidden":
        raise HTTPException(status_code=409, detail="Category is already hidden.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.custom_categories
                SET status = 'hidden', updated_at = NOW()
                WHERE id = %s AND status = 'public'
                RETURNING id
                """,
                (category_id,),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Shared category not found.")
            await aconn.commit()

    hidden = await _fetch_custom_category(category_id)
    return JSONResponse(content={"category": hidden})


@app.post("/admin/dynamic-prompts/categories/{category_id}/restore")
async def admin_restore_dynamic_prompt_category(category_id: str, user: dict = Depends(require_admin)):
    existing = await _fetch_custom_category(category_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found.")
    if existing["status"] != "hidden":
        raise HTTPException(status_code=409, detail="Only hidden categories can be restored.")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE dynamic_prompt.custom_categories
                SET status = 'public', updated_at = NOW()
                WHERE id = %s AND status = 'hidden'
                RETURNING id
                """,
                (category_id,),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Hidden category not found.")
            await aconn.commit()

    restored = await _fetch_custom_category(category_id)
    return JSONResponse(content={"category": restored})


async def resolve_civitai_model_link(version_id: int) -> str:
    """Resolve a CivitAI model page URL from a model version id."""
    if not _has_civitai_version_id(version_id):
        raise HTTPException(status_code=404, detail="Manual LoRAs do not have a CivitAI link")

    cached = civitai_link_cache.get(version_id)
    if cached:
        return cached

    if not session:
        raise HTTPException(status_code=503, detail="CivitAI resolver unavailable")

    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to resolve CivitAI version {version_id}: HTTP {resp.status}",
                )
            data = await resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error resolving CivitAI link for version {version_id}: {e}")
        raise HTTPException(status_code=502, detail="Failed to resolve CivitAI link")

    model_id = data.get("modelId") or (data.get("model") or {}).get("id")
    if not model_id:
        raise HTTPException(status_code=502, detail="CivitAI response missing modelId")

    resolved = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
    civitai_link_cache[version_id] = resolved
    return resolved


@app.get("/admin/civitai-link/{version_id}")
async def get_admin_civitai_link(version_id: int, user: dict = Depends(require_admin)):
    """Return the correct CivitAI model page URL for a model version id. Admin only."""
    url = await resolve_civitai_model_link(version_id)
    return {"url": url}


@app.get("/get_my_lora_suggestions/")
async def get_my_lora_suggestions(status: str = "pending", user: dict = Depends(require_auth)):
    """Get LoRA suggestions for the current user."""
    requestor_candidates = get_lora_requestor_candidates(user)
    if not requestor_candidates:
        raise HTTPException(status_code=400, detail="No requestor id available")

    status_norm = (status or "pending").strip().lower()

    where_clause = "WHERE requestor = ANY(%s)"
    params: list[Any] = [requestor_candidates]
    if status_norm != "all":
        where_clause += " AND status = %s"
        params.append(status_norm)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                  SELECT version_id, name, version, status, requestor, 
                      is_nsfw, is_minor, preview_image, base_model,
                      error_message, last_updated_date,
                      CASE
                          WHEN status = 'rejected'
                          THEN COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day')
                          ELSE NULL
                      END AS rerequest_available_at,
                      CASE
                          WHEN status = 'rejected'
                          THEN GREATEST(
                              0,
                              CEIL(EXTRACT(EPOCH FROM ((COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day')) - NOW())))::INTEGER
                          )
                          ELSE 0
                      END AS cooldown_seconds_remaining
                FROM lora_suggestions
                """ + where_clause + """
                ORDER BY name
                """,
                (LORA_REREQUEST_COOLDOWN_DAYS, LORA_REREQUEST_COOLDOWN_DAYS, *params),
            )
            columns = [desc[0] for desc in acur.description]
            rows = await acur.fetchall()
            result = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                row_dict['id'] = row_dict['version_id']
                row_dict['submitted_by'] = row_dict.get('requestor', '')
                row_dict['image_url'] = row_dict.get('preview_image', '')
                result.append(row_dict)

    json_compatible_result = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_result)


@app.get("/get_all_suggestion_statuses/")
async def get_all_suggestion_statuses():
    """Get all lora suggestion version_ids grouped by status (public, lightweight)."""
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT version_id,
                       status,
                       last_updated_date,
                       CASE
                           WHEN status = 'rejected'
                           THEN COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day')
                           ELSE NULL
                       END AS rerequest_available_at,
                       CASE
                           WHEN status = 'rejected'
                           THEN GREATEST(
                               0,
                               CEIL(EXTRACT(EPOCH FROM ((COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day')) - NOW())))::INTEGER
                           )
                           ELSE 0
                       END AS cooldown_seconds_remaining
                FROM lora_suggestions
                WHERE status IN ('rejected', 'approved', 'pending', 'downloading')
                """,
                (LORA_REREQUEST_COOLDOWN_DAYS, LORA_REREQUEST_COOLDOWN_DAYS),
            )
            rows = await acur.fetchall()

    result: dict = {"rejected": [], "approved": [], "pending": [], "downloading": [], "rejected_cooldowns": {}}
    for version_id, status, last_updated_date, rerequest_available_at, cooldown_seconds_remaining in rows:
        key = (status or "").strip().lower()
        if key in result:
            result[key].append(version_id)
            if key == "rejected":
                result["rejected_cooldowns"][str(version_id)] = {
                    "last_updated_date": last_updated_date,
                    "rerequest_available_at": rerequest_available_at,
                    "cooldown_seconds_remaining": cooldown_seconds_remaining,
                }
    return JSONResponse(content=jsonable_encoder(result))


@app.post("/cancel_lora_suggestion/{suggestion_id}/")
async def cancel_lora_suggestion(suggestion_id: int, user: dict = Depends(require_auth)):
    """Cancel a pending LoRA suggestion for the current user."""
    requestor_candidates = set(get_lora_requestor_candidates(user))
    if not requestor_candidates:
        raise HTTPException(status_code=400, detail="No requestor id available")

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT version_id, name, status, requestor
                FROM lora_suggestions
                WHERE version_id = %s
                """,
                (suggestion_id,),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Suggestion not found")

            _, name, status, row_requestor = row
            if str(row_requestor) not in requestor_candidates:
                raise HTTPException(status_code=403, detail="You can only cancel your own suggestions")

            if status != 'pending':
                raise HTTPException(status_code=409, detail=f"Cannot cancel suggestion in status '{status}'")

            await acur.execute(
                """
                UPDATE lora_suggestions
                SET status = 'cancelled', last_updated_date = NOW()
                WHERE version_id = %s
                """,
                (suggestion_id,),
            )
            await aconn.commit()

    return {
        "status": "success",
        "message": f"Suggestion '{name}' cancelled"
    }


@app.get("/get_lora_suggestions/")
async def get_lora_suggestions(status: str = "pending", user: dict = Depends(require_admin)):
    """Get LoRA suggestions by status. Admin only."""
    status_norm = (status or "pending").strip().lower()

    where_clause = ""
    params: tuple = ()
    if status_norm != "all":
        where_clause = "WHERE status = %s"
        params = (status_norm,)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                  SELECT version_id, name, version, status, requestor, 
                      is_nsfw, is_minor, preview_image, base_model,
                      error_message, last_updated_date,
                      CASE
                          WHEN status = 'rejected'
                          THEN COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day')
                          ELSE NULL
                      END AS rerequest_available_at,
                      CASE
                          WHEN status = 'rejected'
                          THEN GREATEST(
                              0,
                              CEIL(EXTRACT(EPOCH FROM ((COALESCE(last_updated_date, NOW()) + (%s * INTERVAL '1 day')) - NOW())))::INTEGER
                          )
                          ELSE 0
                      END AS cooldown_seconds_remaining
                FROM lora_suggestions
                """ + where_clause + """
                ORDER BY name
                """,
                (LORA_REREQUEST_COOLDOWN_DAYS, LORA_REREQUEST_COOLDOWN_DAYS, *params),
            )
            columns = [desc[0] for desc in acur.description]
            rows = await acur.fetchall()
            # Add 'id' field for frontend compatibility (use version_id)
            result = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                row_dict['id'] = row_dict['version_id']  # Use version_id as id
                row_dict['submitted_by'] = row_dict.get('requestor', '')
                row_dict['image_url'] = row_dict.get('preview_image', '')
                result.append(row_dict)
    
    json_compatible_result = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_result)


class LoraToggleRequest(BaseModel):
    is_active: Optional[bool] = None
    is_nsfw: Optional[bool] = None
    name: Optional[str] = None
    trigger_words: Optional[Any] = None


def _normalize_trigger_words_input(raw_value: Any) -> List[str]:
    """Normalize trigger words to a de-duplicated string list."""
    if raw_value is None:
        return []

    values: List[Any]
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            values = []
        elif candidate.startswith("["):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    values = parsed
                else:
                    values = re.split(r"[\n,]", candidate)
            except Exception:
                values = re.split(r"[\n,]", candidate)
        else:
            values = re.split(r"[\n,]", candidate)
    elif isinstance(raw_value, list):
        values = raw_value
    else:
        raise HTTPException(
            status_code=400,
            detail="trigger_words must be an array or a comma/newline-separated string",
        )

    normalized: List[str] = []
    seen = set()
    for item in values:
        if item is None:
            continue
        cleaned = str(item).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _sanitize_lora_filename(name: str) -> str:
    sanitized = name.replace("'", "")
    sanitized = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in sanitized).strip()
    return sanitized or "untitled"


def _coerce_lora_json_field(raw_value: Any) -> Any:
    if raw_value is None:
        return None
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if isinstance(raw_value, memoryview):
        raw_value = raw_value.tobytes()
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="ignore")
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _validate_safetensors_header_bytes(raw_bytes: bytes, max_header_bytes: int = ADMIN_LORA_HEADER_MAX_BYTES) -> Dict[str, Any]:
    if len(raw_bytes) < 8:
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header is missing.")

    header_length = int.from_bytes(raw_bytes[:8], "little")
    if header_length <= 0:
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header length is invalid.")
    if header_length > max_header_bytes:
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header is too large.")

    required_length = 8 + header_length
    if len(raw_bytes) < required_length:
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header is truncated.")

    try:
        header = json.loads(raw_bytes[8:required_length].decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid safetensors file: header JSON is unreadable ({exc}).")

    if not isinstance(header, dict):
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header JSON must be an object.")

    return header


async def _validate_safetensors_upload(file: UploadFile) -> Dict[str, Any]:
    await file.seek(0)
    prefix = await file.read(8)
    if len(prefix) < 8:
        await file.seek(0)
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header is missing.")

    header_length = int.from_bytes(prefix, "little")
    if header_length <= 0:
        await file.seek(0)
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header length is invalid.")
    if header_length > ADMIN_LORA_HEADER_MAX_BYTES:
        await file.seek(0)
        raise HTTPException(status_code=400, detail="Invalid safetensors file: header is too large.")

    header_bytes = await file.read(header_length)
    try:
        return _validate_safetensors_header_bytes(prefix + header_bytes)
    finally:
        await file.seek(0)


async def _stream_upload_to_temp_file(file: UploadFile, temp_dir: str, max_bytes: int) -> Tuple[str, str, int]:
    os.makedirs(temp_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="manual-lora-", suffix=".tmp", dir=temp_dir)
    sha256 = hashlib.sha256()
    total_bytes = 0

    try:
        await file.seek(0)
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"LoRA file is too large. Limit is {max_bytes // (1024 * 1024)} MB.",
                    )
                sha256.update(chunk)
                handle.write(chunk)
        if total_bytes <= 0:
            raise HTTPException(status_code=400, detail="Empty safetensors upload.")
        return temp_path, sha256.hexdigest().upper(), total_bytes
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    finally:
        await file.seek(0)


def _build_lora_storage_path(name: str, version: str, base_model: str, storage_identity: Optional[Any] = None) -> str:
    sanitized_name = _sanitize_lora_filename(name)
    sanitized_version = _sanitize_lora_filename(version)
    identity_suffix = ""
    if storage_identity is not None:
        identity_suffix = f"-{_sanitize_lora_filename(str(storage_identity))}"
    filename = f"{sanitized_name}-{sanitized_version}{identity_suffix}.safetensors"
    base_model_folder = os.path.join(LORAS_FOLDER, _sanitize_lora_filename(base_model))
    lora_folder = os.path.join(base_model_folder, sanitized_name)
    return os.path.join(lora_folder, filename)


def _build_manual_lora_download_url(name: str, version: str, base_model: str, storage_identity: Optional[Any] = None) -> str:
    sanitized_name = _sanitize_lora_filename(name)
    sanitized_version = _sanitize_lora_filename(version)
    sanitized_base_model = _sanitize_lora_filename(base_model)
    identity_suffix = ""
    if storage_identity is not None:
        identity_suffix = f"/{_sanitize_lora_filename(str(storage_identity))}"
    return f"manual-upload://{sanitized_base_model}/{sanitized_name}/{sanitized_version}{identity_suffix}"


async def _allocate_manual_lora_version_id(acur) -> int:
    await acur.execute("SELECT pg_advisory_xact_lock(%s)", (ADMIN_LORA_VERSION_ID_LOCK_KEY,))
    await acur.execute(
        "SELECT COALESCE(MIN(version_id), 0) FROM lora_metadata WHERE version_id IS NOT NULL"
    )
    row = await acur.fetchone()
    lowest_version_id = int(row[0] or 0)
    return lowest_version_id - 1 if lowest_version_id <= 0 else -1


async def _find_existing_lora_by_sha256(acur, sha256_hash: str) -> Optional[Dict[str, Any]]:
    await acur.execute(
        "SELECT id, name, version, hashes FROM lora_metadata WHERE hashes IS NOT NULL"
    )
    rows = await acur.fetchall()

    for lora_id, name, version, hashes in rows:
        parsed_hashes = _coerce_lora_json_field(hashes)
        if not isinstance(parsed_hashes, dict):
            continue
        existing_hash = str(parsed_hashes.get("SHA256") or "").strip().upper()
        if existing_hash and existing_hash == sha256_hash:
            return {
                "id": lora_id,
                "name": name,
                "version": version,
            }
    return None


def _has_civitai_version_id(version_id: Any) -> bool:
    try:
        return int(version_id) > 0
    except Exception:
        return False


@app.patch("/admin/lora/{lora_id}")
async def admin_update_lora(lora_id: int, data: LoraToggleRequest, user: dict = Depends(require_admin)):
    """Update a LoRA's active, NSFW status, name, or trigger words by id. Admin only."""
    updates = []
    params = []
    
    if data.is_active is not None:
        updates.append("is_active = %s")
        params.append(data.is_active)
    
    if data.is_nsfw is not None:
        updates.append("is_nsfw = %s")
        params.append(data.is_nsfw)
    
    if data.name is not None:
        updates.append("name = %s")
        params.append(data.name)

    if data.trigger_words is not None:
        normalized_trigger_words = _normalize_trigger_words_input(data.trigger_words)
        updates.append("trigger_words = %s")
        params.append(json.dumps(normalized_trigger_words))
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    params.append(lora_id)
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                f"""
                UPDATE lora_metadata
                SET {', '.join(updates)}
                WHERE id = %s
                RETURNING id, name, is_active, is_nsfw, trigger_words
                """,
                tuple(params)
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="LoRA not found")
            
            await aconn.commit()
    
    return {
        "status": "success",
        "lora": {
            "id": row[0],
            "name": row[1],
            "is_active": row[2],
            "is_nsfw": row[3],
            "trigger_words": row[4] if len(row) > 4 else None
        }
    }


@app.post("/admin/lora/upload")
async def admin_upload_manual_lora(
    file: UploadFile = File(...),
    preview_image: UploadFile = File(...),
    name: str = Form(...),
    version: str = Form(...),
    base_model: str = Form(...),
    trigger_words: Optional[str] = Form(None),
    creator: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    is_nsfw: bool = Form(False),
    user: dict = Depends(require_admin),
):
    normalized_name = (name or "").strip()
    normalized_version = (version or "").strip()
    normalized_base_model = (base_model or "").strip()
    normalized_creator = (creator or "").strip() or None
    normalized_description = (description or "").strip() or None

    if not normalized_name:
        raise HTTPException(status_code=400, detail="Name is required.")
    if not normalized_version:
        raise HTTPException(status_code=400, detail="Version is required.")
    if not is_supported_lora_base_model(normalized_base_model):
        raise HTTPException(status_code=400, detail="Unsupported base model.")

    original_filename = (file.filename or "").strip()
    if not original_filename.lower().endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Model file must be a .safetensors file.")

    if not preview_image.filename:
        raise HTTPException(status_code=400, detail="Preview image is required for manual uploads.")
    if not preview_image.content_type or not preview_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Preview image must be an image file.")

    normalized_trigger_words = _normalize_trigger_words_input(trigger_words)
    await _validate_safetensors_upload(file)

    preview_content = await preview_image.read()
    if not preview_content:
        raise HTTPException(status_code=400, detail="Preview image upload is empty.")

    try:
        optimized_preview = _process_lora_preview_image(preview_content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process preview image: {exc}")

    temp_directory = os.path.dirname(_build_lora_storage_path(normalized_name, normalized_version, normalized_base_model))
    temp_file_path, sha256_hash, file_size = await _stream_upload_to_temp_file(
        file,
        temp_directory,
        ADMIN_LORA_MAX_UPLOAD_BYTES,
    )
    final_file_path = _build_lora_storage_path(
        normalized_name,
        normalized_version,
        normalized_base_model,
        sha256_hash,
    )
    manual_download_url = _build_manual_lora_download_url(
        normalized_name,
        normalized_version,
        normalized_base_model,
        sha256_hash,
    )
    final_file_written = False
    created_row = None
    columns: List[str] = []

    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                existing_hash_match = await _find_existing_lora_by_sha256(acur, sha256_hash)
                if existing_hash_match:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f'This safetensors file matches the existing LoRA '
                            f'{existing_hash_match["name"]} ({existing_hash_match["version"]}).'
                        ),
                    )

                if os.path.exists(final_file_path):
                    raise HTTPException(
                        status_code=409,
                        detail="A file already exists at the target LoRA path. Rename the LoRA or clean up the existing file first.",
                    )

                version_id = await _allocate_manual_lora_version_id(acur)
                os.replace(temp_file_path, final_file_path)
                final_file_written = True

                await acur.execute(
                    """
                    INSERT INTO lora_metadata (
                        name,
                        version,
                        base_model,
                        download_url,
                        is_nsfw,
                        is_minor,
                        creator,
                        description,
                        version_description,
                        tags,
                        who_added,
                        status,
                        trigger_words,
                        hashes,
                        image_url,
                        file_path,
                        version_id,
                        image_blob
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                    """,
                    (
                        normalized_name,
                        normalized_version,
                        normalized_base_model,
                        manual_download_url,
                        bool(is_nsfw),
                        False,
                        normalized_creator,
                        normalized_description,
                        None,
                        json.dumps([]),
                        get_primary_lora_requestor(user),
                        "downloaded",
                        json.dumps(normalized_trigger_words),
                        json.dumps({
                            "SHA256": sha256_hash,
                            "size_bytes": file_size,
                            "source": "manual-upload",
                        }),
                        None,
                        final_file_path,
                        version_id,
                        optimized_preview,
                    ),
                )
                inserted_row = await acur.fetchone()
                if not inserted_row:
                    raise HTTPException(status_code=500, detail="Failed to create LoRA record.")

                lora_id = inserted_row[0]
                image_url = f"/lora-image/{lora_id}?v={int(time.time())}"

                await acur.execute(
                    """
                    UPDATE lora_metadata
                    SET image_url = %s
                    WHERE id = %s
                    RETURNING id, name, version, base_model, download_url, is_nsfw, is_minor,
                              creator, description, version_description, tags, who_added, status,
                              trigger_words, date_added, hashes, image_url, file_path, uses,
                              is_active, last_used_date, version_id
                    """,
                    (image_url, lora_id),
                )
                created_row = await acur.fetchone()
                columns = [desc[0] for desc in acur.description]
                await aconn.commit()
    except HTTPException:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if final_file_written and os.path.exists(final_file_path):
            os.remove(final_file_path)
        raise
    except Exception as exc:
        logging.error(f"Manual LoRA upload failed: {exc}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if final_file_written and os.path.exists(final_file_path):
            os.remove(final_file_path)
        raise HTTPException(status_code=500, detail="Failed to upload manual LoRA.")

    lora_payload = dict(zip(columns, created_row)) if created_row else {}
    return {
        "status": "success",
        "lora": jsonable_encoder(lora_payload),
    }


def _process_lora_preview_image(raw_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(raw_bytes))
    image = image.convert("RGB")
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    image.thumbnail((1024, 1024), resample)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=85, method=6)
    return buffer.getvalue()


@app.post("/admin/lora/{lora_id}/image")
async def admin_upload_lora_image(
    lora_id: int,
    file: UploadFile = File(...),
    user: dict = Depends(require_admin)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    try:
        optimized = _process_lora_preview_image(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}")

    cache_bust = int(time.time())
    image_url = f"/lora-image/{lora_id}?v={cache_bust}"

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE lora_metadata
                SET image_url = %s,
                    image_blob = %s
                WHERE id = %s
                RETURNING id, image_url
                """,
                (image_url, optimized, lora_id),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="LoRA not found")
            await aconn.commit()

    return {"status": "success", "image_url": row[1]}


@app.get("/lora-image/{lora_id}")
async def get_lora_image(lora_id: int, w: Optional[int] = Query(None, gt=0, le=2048)):
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT image_blob
                FROM lora_metadata
                WHERE id = %s
                """,
                (lora_id,)
            )
            row = await acur.fetchone()

    if not row or row[0] is None:
        raise HTTPException(status_code=404, detail="Image not found")

    original_bytes = bytes(row[0])
    if not w:
        return Response(content=original_bytes, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})

    with Image.open(io.BytesIO(original_bytes)) as image:
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        if image.width > w:
            ratio = w / float(image.width)
            height = max(1, int(image.height * ratio))
            image = image.resize((w, height), resample)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=85, method=6)
        data = buffer.getvalue()

    return Response(content=data, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/admin/suggestion/{suggestion_id}/approve")
async def admin_approve_suggestion(suggestion_id: int, user: dict = Depends(require_admin)):
    """Approve a LoRA suggestion so it can be downloaded by the downloader service. Admin only."""
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Get the suggestion (use version_id as the identifier)
            await acur.execute(
                """
                SELECT version_id, name, version, is_nsfw, is_minor, preview_image
                FROM lora_suggestions
                WHERE version_id = %s AND status IN ('pending', 'rejected')
                """,
                (suggestion_id,)
            )
            suggestion = await acur.fetchone()
            if not suggestion:
                raise HTTPException(status_code=404, detail="Suggestion not found or already processed")
            
            version_id, name, version, is_nsfw, is_minor, preview_image = suggestion
            
            # Check if LoRA already exists in metadata
            await acur.execute(
                "SELECT name FROM lora_metadata WHERE version_id = %s",
                (version_id,)
            )
            existing = await acur.fetchone()
            if existing:
                # Mark suggestion as duplicate
                await acur.execute(
                    "UPDATE lora_suggestions SET status = 'duplicate', last_updated_date = NOW() WHERE version_id = %s",
                    (suggestion_id,)
                )
                await aconn.commit()
                raise HTTPException(status_code=409, detail="LoRA already exists in database")
            
            # Update suggestion status to 'approved' - the downloader service will handle the rest
            await acur.execute(
                "UPDATE lora_suggestions SET status = 'approved', last_updated_date = NOW() WHERE version_id = %s",
                (suggestion_id,)
            )
            
            await aconn.commit()
    
    return {
        "status": "success",
        "message": f"LoRA '{name}' approved and queued for download"
    }


@app.post("/admin/suggestion/{suggestion_id}/reject")
async def admin_reject_suggestion(suggestion_id: int, user: dict = Depends(require_admin)):
    """Reject a LoRA suggestion. Admin only."""
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                UPDATE lora_suggestions
                SET status = 'rejected',
                    last_updated_date = NOW()
                WHERE version_id = %s AND status = 'pending'
                RETURNING version_id, name
                """,
                (suggestion_id,)
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Suggestion not found or already processed")
            
            await aconn.commit()
    
    return {
        "status": "success",
        "message": f"Suggestion '{row[1]}' rejected"
    }


# ============================================
# DOWNLOADER STATUS ENDPOINTS
# ============================================

DOWNLOADER_SERVICE_URL = os.environ.get("DOWNLOADER_SERVICE_URL", "http://localhost:9002")


@app.get("/admin/downloader-status")
async def get_downloader_status(user: dict = Depends(require_admin)):
    """Get the current status of the LoRA downloader service. Admin only."""
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                # Check if anything is currently downloading
                await acur.execute("""
                    SELECT name, version, last_updated_date
                    FROM lora_suggestions
                    WHERE status = 'downloading'
                    LIMIT 1
                """)
                downloading = await acur.fetchone()
                
                # Count approved (pending download)
                await acur.execute("""
                    SELECT COUNT(*) FROM lora_suggestions WHERE status = 'approved'
                """)
                approved_count = (await acur.fetchone())[0]
                
                # Get last processed
                await acur.execute("""
                    SELECT name, version, status, error_message, last_updated_date
                    FROM lora_suggestions
                    WHERE status IN ('downloaded', 'failed')
                    ORDER BY last_updated_date DESC
                    LIMIT 1
                """)
                last_processed = await acur.fetchone()
                
                if downloading:
                    return {
                        "status": "downloading",
                        "current_lora": f"{downloading[0]} v{downloading[1]}",
                        "updated_at": downloading[2].isoformat() if downloading[2] else None,
                        "approved_count": approved_count,
                        "last_processed": None
                    }
                else:
                    return {
                        "status": "idle",
                        "current_lora": None,
                        "approved_count": approved_count,
                        "last_processed": {
                            "name": f"{last_processed[0]} v{last_processed[1]}" if last_processed else None,
                            "status": last_processed[2] if last_processed else None,
                            "error_message": last_processed[3] if last_processed else None,
                            "updated_at": last_processed[4].isoformat() if last_processed and last_processed[4] else None
                        } if last_processed else None,
                        "updated_at": last_processed[4].isoformat() if last_processed and last_processed[4] else None
                    }
    except Exception as e:
        logging.error(f"Error fetching downloader status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/download-history")
async def get_download_history(limit: int = 20, user: dict = Depends(require_admin)):
    """Get recent download history from lora_suggestions. Admin only."""
    try:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute("""
                    SELECT name, version, status, error_message, version_id, last_updated_date
                    FROM lora_suggestions
                    WHERE status IN ('downloaded', 'failed', 'downloading')
                    ORDER BY last_updated_date DESC
                    LIMIT %s
                """, (limit,))
                rows = await acur.fetchall()
                
                history = [{
                    "lora_name": row[0],
                    "version": row[1],
                    "status": row[2],
                    "error_message": row[3],
                    "version_id": row[4],
                    "downloaded_at": row[5].isoformat() if row[5] else None
                } for row in rows]
                
                return history
    except Exception as e:
        logging.error(f"Error fetching download history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/trigger-download")
async def trigger_download(user: dict = Depends(require_admin)):
    """Trigger the downloader service to check for new approved LoRAs. Admin only."""
    try:
        async with aiohttp.ClientSession() as trigger_session:
            async with trigger_session.post(f"{DOWNLOADER_SERVICE_URL}/trigger", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {"status": "success", "message": result.get("message", "Download triggered")}
                elif resp.status == 409:
                    result = await resp.json()
                    return {"status": "busy", "message": result.get("message", "Download already in progress")}
                else:
                    raise HTTPException(status_code=resp.status, detail="Failed to trigger download")
    except aiohttp.ClientError as e:
        logging.error(f"Error triggering downloader: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Downloader service is not reachable. Make sure lora_downloader_service.py is running."
        )


# Azure health check, return 200
@app.get("/health_check")
async def health_check():
    return {"status": 200}

# New endpoint to cancel a pending job by ID
@app.delete("/cancel_job/{job_id}/")
async def cancel_job(job_id: str):
    """Cancel a job by deleting it from the queue if it is still pending."""
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Attempt to delete only if job is still pending, and get user/credit info for refund
            await acur.execute(
                """
                DELETE FROM generation_queue
                WHERE id = %s AND status = 'pending'
                RETURNING id, user_id, queue_type, credit_cost;
                """,
                (job_id,),
            )
            deleted = await acur.fetchone()
            if deleted:
                job_id_deleted, user_id, queue_type, credit_cost = deleted
                
                # Refund credits if this was a priority job with credits deducted
                credits_refunded = 0
                if queue_type == "priority" and credit_cost > 0 and user_id:
                    await acur.execute(
                        "SELECT * FROM refund_credits(%s, %s, %s, %s)",
                        (user_id, credit_cost, job_id_deleted, "Generation cancelled by user")
                    )
                    refund_result = await acur.fetchone()
                    if refund_result and refund_result[0]:  # success = True
                        credits_refunded = credit_cost
                
                await aconn.commit()
                return JSONResponse(content={
                    "status": "success", 
                    "job_id": job_id,
                    "credits_refunded": credits_refunded
                })

            # If not deleted, check if it exists and report why it can't be cancelled
            await acur.execute(
                """
                SELECT status FROM generation_queue WHERE id = %s;
                """,
                (job_id,),
            )
            row = await acur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            else:
                raise HTTPException(status_code=409, detail=f"Cannot cancel job in status '{row[0]}'")


# ============================================================================
# IMAGE HISTORY SYNC ENDPOINTS
# ============================================================================

class SyncImageRequest(BaseModel):
    image_uuid: str
    prompt: Optional[str] = None
    prompt_summary: Optional[str] = None
    prompt_template: Optional[str] = None
    negative_prompt: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    cfg: Optional[float] = None
    width: int
    height: int
    aspect_ratio: str
    is_favorite: bool = False
    sync_priority: int = 0
    loras: Optional[List[Any]] = []
    regional_prompting: Optional[dict] = None
    tags: Optional[List[str]] = []
    image_blob: str  # Base64 encoded image


class SyncTagsRequest(BaseModel):
    tags: List[dict]


class LoraPreferenceItem(BaseModel):
    version_id: int
    is_favorite: Optional[bool] = None
    last_used_at: Optional[datetime] = None


class LoraPreferencesSyncRequest(BaseModel):
    preferences: List[LoraPreferenceItem]


class RegionalPromptPresetRegionItem(BaseModel):
    id: str
    prompt: str = ""
    negative_prompt: str = ""
    x: float = 0
    y: float = 0
    width: float = 0.25
    height: float = 0.25
    denoise_strength: float = 1.0
    feather: int = 2
    opacity: float = 1.0
    inherit_base_prompt: bool = False


class RegionalPromptPresetItem(BaseModel):
    id: Optional[str] = None
    name: str
    regions: List[RegionalPromptPresetRegionItem]


class RegionalPromptPresetsSyncRequest(BaseModel):
    presets: List[RegionalPromptPresetItem]


@app.get("/history/sync/status")
async def get_sync_status(user: dict = Depends(require_auth)):
    """Get the current sync status for the user."""
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Count synced images
            await acur.execute(
                "SELECT COUNT(*) FROM user_synced_images WHERE user_id = %s",
                (user_id,)
            )
            count_row = await acur.fetchone()
            images_in_cloud = count_row[0] if count_row else 0
            
            # Get last sync time (most recent updated_at)
            await acur.execute(
                "SELECT MAX(updated_at) FROM user_synced_images WHERE user_id = %s",
                (user_id,)
            )
            time_row = await acur.fetchone()
            last_sync_time = time_row[0].isoformat() if time_row and time_row[0] else None
            
            # Get list of synced UUIDs
            await acur.execute(
                "SELECT image_uuid FROM user_synced_images WHERE user_id = %s",
                (user_id,)
            )
            uuid_rows = await acur.fetchall()
            synced_uuids = [row[0] for row in uuid_rows]
    
    return {
        "images_in_cloud": images_in_cloud,
        "quota_limit": 1000,
        "last_sync_time": last_sync_time,
        "synced_uuids": synced_uuids
    }


@app.post("/history/sync/image")
async def sync_image(request: SyncImageRequest, user: dict = Depends(require_auth)):
    """Sync a single image to the cloud."""
    user_id = user["user_id"]
    
    # Check quota
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                "SELECT COUNT(*) FROM user_synced_images WHERE user_id = %s",
                (user_id,)
            )
            count_row = await acur.fetchone()
            current_count = count_row[0] if count_row else 0
            
            # Check if this image already exists (update case)
            await acur.execute(
                "SELECT id FROM user_synced_images WHERE user_id = %s AND image_uuid = %s",
                (user_id, request.image_uuid)
            )
            existing = await acur.fetchone()
            
            if not existing and current_count >= 1000:
                raise HTTPException(status_code=400, detail="Sync quota exceeded (1000 images max)")
            
            # Decode base64 blob
            try:
                image_blob = base64.b64decode(request.image_blob)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
            
            # Upsert the image
            await acur.execute(
                """
                INSERT INTO user_synced_images (
                    user_id, image_uuid, prompt, prompt_summary, prompt_template, negative_prompt,
                    model, seed, cfg, width, height, aspect_ratio,
                    is_favorite, sync_priority, loras, regional_prompting, tags, image_blob, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (user_id, image_uuid) DO UPDATE SET
                    prompt = EXCLUDED.prompt,
                    prompt_summary = EXCLUDED.prompt_summary,
                    prompt_template = EXCLUDED.prompt_template,
                    negative_prompt = EXCLUDED.negative_prompt,
                    model = EXCLUDED.model,
                    seed = EXCLUDED.seed,
                    cfg = EXCLUDED.cfg,
                    width = EXCLUDED.width,
                    height = EXCLUDED.height,
                    aspect_ratio = EXCLUDED.aspect_ratio,
                    is_favorite = EXCLUDED.is_favorite,
                    sync_priority = EXCLUDED.sync_priority,
                    loras = EXCLUDED.loras,
                    regional_prompting = EXCLUDED.regional_prompting,
                    tags = EXCLUDED.tags,
                    image_blob = EXCLUDED.image_blob,
                    updated_at = NOW()
                """,
                (
                    user_id, request.image_uuid, request.prompt, request.prompt_summary,
                    request.prompt_template, request.negative_prompt, request.model, request.seed, request.cfg,
                    request.width, request.height, request.aspect_ratio,
                    request.is_favorite, request.sync_priority,
                    json.dumps(request.loras or []),
                    json.dumps(request.regional_prompting) if request.regional_prompting is not None else None,
                    json.dumps(request.tags or []),
                    image_blob
                )
            )
            await aconn.commit()
    
    return {"success": True, "image_uuid": request.image_uuid}


class UpdateImageMetadataRequest(BaseModel):
    """Request to update just the metadata (tags, favorite) of a synced image."""
    tags: Optional[List[str]] = None
    is_favorite: Optional[bool] = None


@app.patch("/history/sync/image/{image_uuid}")
async def update_image_metadata(
    image_uuid: str, 
    request: UpdateImageMetadataRequest, 
    user: dict = Depends(require_auth)
):
    """Update metadata (tags, favorite status) for a synced image without re-uploading the blob."""
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Check if image exists
            await acur.execute(
                "SELECT id FROM user_synced_images WHERE user_id = %s AND image_uuid = %s",
                (user_id, image_uuid)
            )
            existing = await acur.fetchone()
            
            if not existing:
                raise HTTPException(status_code=404, detail="Image not found in sync")
            
            # Build update query dynamically based on what's provided
            updates = []
            params = []
            
            if request.tags is not None:
                updates.append("tags = %s")
                params.append(json.dumps(request.tags))
            
            if request.is_favorite is not None:
                updates.append("is_favorite = %s")
                params.append(request.is_favorite)
            
            if updates:
                updates.append("updated_at = NOW()")
                params.extend([user_id, image_uuid])
                
                query = f"""
                    UPDATE user_synced_images 
                    SET {', '.join(updates)}
                    WHERE user_id = %s AND image_uuid = %s
                """
                await acur.execute(query, tuple(params))
                await aconn.commit()
    
    return {"success": True, "image_uuid": image_uuid}


@app.get("/history/sync/images")
async def get_synced_images(include_blobs: bool = True, user: dict = Depends(require_auth)):
    """Get all synced images for the user."""
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            if include_blobs:
                query = """
                SELECT image_uuid, prompt, prompt_summary, prompt_template, negative_prompt, model,
                       seed, cfg, width, height, aspect_ratio, is_favorite,
                       sync_priority, loras, regional_prompting, tags, image_blob, created_at
                FROM user_synced_images
                WHERE user_id = %s
                ORDER BY sync_priority DESC, created_at DESC
                """
            else:
                query = """
                SELECT image_uuid, prompt, prompt_summary, prompt_template, negative_prompt, model,
                       seed, cfg, width, height, aspect_ratio, is_favorite,
                       sync_priority, loras, regional_prompting, tags, created_at
                FROM user_synced_images
                WHERE user_id = %s
                ORDER BY sync_priority DESC, created_at DESC
                """
            await acur.execute(query, (user_id,))
            rows = await acur.fetchall()
    
    images = []
    for row in rows:
        image_blob_b64 = None
        created_at = row[17] if include_blobs else row[16]
        if include_blobs and row[16]:
            image_blob_b64 = base64.b64encode(row[16]).decode('utf-8')
        
        images.append({
            "image_uuid": row[0],
            "prompt": row[1],
            "prompt_summary": row[2],
            "prompt_template": row[3],
            "negative_prompt": row[4],
            "model": row[5],
            "seed": row[6],
            "cfg": float(row[7]) if row[7] else None,
            "width": row[8],
            "height": row[9],
            "aspect_ratio": row[10],
            "is_favorite": row[11],
            "sync_priority": row[12],
            "loras": row[13] if row[13] else [],
            "regional_prompting": row[14] if row[14] else {"enabled": False, "regions": []},
            "tags": row[15] if row[15] else [],
            "image_blob": image_blob_b64,
            "created_at": created_at.isoformat() if created_at else None
        })
    
    return images


@app.delete("/history/sync/image/{image_uuid}")
async def delete_synced_image(image_uuid: str, user: dict = Depends(require_auth)):
    """Remove an image from cloud sync."""
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                "DELETE FROM user_synced_images WHERE user_id = %s AND image_uuid = %s RETURNING id",
                (user_id, image_uuid)
            )
            deleted = await acur.fetchone()
            await aconn.commit()
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Image not found in sync")
    
    return {"success": True, "deleted": image_uuid}


@app.post("/history/sync/tags")
async def sync_tags(request: SyncTagsRequest, user: dict = Depends(require_auth)):
    """Sync user tags to the cloud.
    
    Uses upsert on primary key (id) to handle the case where a tag
    already exists. The unique constraint on (user_id, name) prevents
    duplicate tag names per user - if a name conflict occurs, we update
    the existing tag's color.
    """
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            for tag in request.tags:
                tag_id = tag.get("id")
                tag_name = tag.get("name")
                tag_color = tag.get("color")
                
                # First, check if a tag with this name already exists for this user
                await acur.execute(
                    "SELECT id FROM user_tags WHERE user_id = %s AND name = %s",
                    (user_id, tag_name)
                )
                existing = await acur.fetchone()
                
                if existing:
                    # Tag with this name exists - update it (use existing id)
                    await acur.execute(
                        "UPDATE user_tags SET color = %s WHERE id = %s",
                        (tag_color, existing[0])
                    )
                else:
                    # No existing tag with this name - insert new
                    await acur.execute(
                        """
                        INSERT INTO user_tags (id, user_id, name, color, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            color = EXCLUDED.color
                        """,
                        (tag_id, user_id, tag_name, tag_color)
                    )
            await aconn.commit()
    
    return {"success": True, "synced_count": len(request.tags)}


@app.get("/history/sync/tags")
async def get_synced_tags(user: dict = Depends(require_auth)):
    """Get user's synced tags."""
    user_id = user["user_id"]
    
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                "SELECT id, name, color, created_at FROM user_tags WHERE user_id = %s ORDER BY created_at",
                (user_id,)
            )
            rows = await acur.fetchall()
    
    return [
        {
            "id": row[0],
            "name": row[1],
            "color": row[2],
            "created_at": row[3].isoformat() if row[3] else None
        }
        for row in rows
    ]


@app.delete("/history/sync/tags/{tag_id}")
async def delete_synced_tag(tag_id: str, user: dict = Depends(require_auth)):
    """Delete a synced tag and remove it from synced images."""
    user_id = user["user_id"]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                "DELETE FROM user_tags WHERE user_id = %s AND id = %s RETURNING id",
                (user_id, tag_id)
            )
            deleted = await acur.fetchone()
            if not deleted:
                raise HTTPException(status_code=404, detail="Tag not found")

            await acur.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = 'user_synced_images'
                  AND column_name = 'tags'
                  AND table_schema = current_schema()
                LIMIT 1
                """
            )
            type_row = await acur.fetchone()
            tags_type = type_row[0] if type_row else 'jsonb'

            if tags_type == 'json':
                await acur.execute(
                    """
                    UPDATE user_synced_images
                    SET tags = COALESCE(
                        (
                            SELECT jsonb_agg(elem)
                            FROM jsonb_array_elements_text(COALESCE(tags::jsonb, '[]'::jsonb)) elem
                            WHERE elem <> %s
                        ),
                        '[]'::jsonb
                    )::json
                    WHERE user_id = %s
                    """,
                    (tag_id, user_id)
                )
            else:
                await acur.execute(
                    """
                    UPDATE user_synced_images
                    SET tags = COALESCE(
                        (
                            SELECT jsonb_agg(elem)
                            FROM jsonb_array_elements_text(COALESCE(tags, '[]'::jsonb)) elem
                            WHERE elem <> %s
                        ),
                        '[]'::jsonb
                    )
                    WHERE user_id = %s
                    """,
                    (tag_id, user_id)
                )

            await aconn.commit()

    return {"success": True, "deleted": tag_id}


# LORA PREFERENCES SYNC ENDPOINTS
@app.get("/lora/preferences")
async def get_lora_preferences(user: dict = Depends(require_auth)):
    """Get user's LoRA favorites and last-used timestamps."""
    user_id = user["user_id"]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT version_id, is_favorite, last_used_at, updated_at
                FROM user_lora_preferences
                WHERE user_id = %s
                """,
                (user_id,)
            )
            rows = await acur.fetchall()

    return [
        {
            "version_id": row[0],
            "is_favorite": row[1],
            "last_used_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
        }
        for row in rows
    ]


@app.post("/lora/preferences")
async def sync_lora_preferences(request: LoraPreferencesSyncRequest, user: dict = Depends(require_auth)):
    """Sync user's LoRA preferences (favorites and last-used)."""
    user_id = user["user_id"]
    prefs = request.preferences or []

    if len(prefs) == 0:
        return {"success": True, "synced_count": 0}

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            for pref in prefs:
                if pref is None:
                    continue
                if pref.is_favorite is None and pref.last_used_at is None:
                    continue
                await acur.execute(
                    """
                    INSERT INTO user_lora_preferences (
                        user_id, version_id, is_favorite, last_used_at, created_at, updated_at
                    ) VALUES (
                        %s, %s, COALESCE(%s, FALSE), %s, NOW(), NOW()
                    )
                    ON CONFLICT (user_id, version_id) DO UPDATE SET
                        is_favorite = COALESCE(EXCLUDED.is_favorite, user_lora_preferences.is_favorite),
                        last_used_at = COALESCE(EXCLUDED.last_used_at, user_lora_preferences.last_used_at),
                        updated_at = NOW()
                    """,
                    (user_id, pref.version_id, pref.is_favorite, pref.last_used_at)
                )
            await aconn.commit()

    return {"success": True, "synced_count": len(prefs)}


@app.get("/regional-presets")
async def get_regional_prompt_presets(user: dict = Depends(require_auth)):
    """Get account-backed regional prompt presets for the current user."""
    user_id = user["user_id"]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT id, name, regions, updated_at
                FROM user_regional_prompt_presets
                WHERE user_id = %s
                ORDER BY updated_at DESC, created_at DESC
                """,
                (user_id,)
            )
            rows = await acur.fetchall()

    return [
        {
            "id": str(row[0]),
            "name": row[1],
            "regions": row[2] if isinstance(row[2], list) else [],
            "updated_at": row[3].isoformat() if row[3] else None,
        }
        for row in rows
    ]


@app.post("/regional-presets")
async def sync_regional_prompt_presets(
    request: RegionalPromptPresetsSyncRequest,
    user: dict = Depends(require_auth),
):
    """Replace the current user's preset set with the provided list."""
    user_id = user["user_id"]
    incoming_presets = request.presets or []

    # Keep payloads bounded.
    incoming_presets = incoming_presets[:100]
    sanitized_presets: List[dict] = []

    for preset in incoming_presets:
        if preset is None:
            continue

        name = (preset.name or "").strip()
        if not name:
            continue
        name = name[:120]

        regions = []
        for region in (preset.regions or [])[:32]:
            width = min(1.0, max(0.05, float(region.width)))
            height = min(1.0, max(0.05, float(region.height)))
            x = min(1.0, max(0.0, float(region.x)))
            y = min(1.0, max(0.0, float(region.y)))
            if x + width > 1.0:
                x = max(0.0, 1.0 - width)
            if y + height > 1.0:
                y = max(0.0, 1.0 - height)

            regions.append(
                {
                    "id": str(region.id or f"{int(time.time() * 1000)}-{secrets.randbelow(1000)}"),
                    "prompt": (region.prompt or "").strip(),
                    "negative_prompt": (region.negative_prompt or "").strip(),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "denoise_strength": min(1.0, max(0.0, float(region.denoise_strength))),
                    "feather": min(96, max(0, int(region.feather))),
                    "opacity": min(1.0, max(0.0, float(region.opacity))),
                    "inherit_base_prompt": bool(region.inherit_base_prompt),
                }
            )

        if len(regions) == 0:
            continue

        preset_id = None
        if preset.id:
            try:
                preset_id = uuid.UUID(str(preset.id))
            except Exception:
                preset_id = None

        sanitized_presets.append(
            {
                "id": preset_id,
                "name": name,
                "regions": regions,
            }
        )

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                "DELETE FROM user_regional_prompt_presets WHERE user_id = %s",
                (user_id,)
            )

            for preset in sanitized_presets:
                await acur.execute(
                    """
                    INSERT INTO user_regional_prompt_presets
                        (id, user_id, name, regions, created_at, updated_at)
                    VALUES
                        (COALESCE(%s, gen_random_uuid()), %s, %s, %s::jsonb, NOW(), NOW())
                    """,
                    (
                        preset["id"],
                        user_id,
                        preset["name"],
                        json.dumps(preset["regions"]),
                    ),
                )

            await aconn.commit()

    return {"success": True, "synced_count": len(sanitized_presets)}


# ============================================
# APRIL FOOLS - RING COLLECTION MINI-GAME
# ============================================

class RingSubmission(BaseModel):
    rings: int


@app.post("/april-fools/rings")
async def submit_rings(data: RingSubmission, user: dict = Depends(require_auth)):
    """Submit collected rings for the leaderboard. Authenticated users only."""
    if data.rings < 1 or data.rings > 1000:
        raise HTTPException(status_code=400, detail="Invalid ring count")

    user_id = user["user_id"]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                INSERT INTO april_fools_rings (user_id, rings_collected, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (user_id) DO UPDATE
                SET rings_collected = april_fools_rings.rings_collected + EXCLUDED.rings_collected,
                    updated_at = NOW()
                RETURNING rings_collected
                """,
                (user_id, data.rings),
            )
            result = await acur.fetchone()
            await aconn.commit()

    return {"status": "success", "total_rings": result[0] if result else data.rings}


@app.get("/april-fools/ring-leaderboard")
async def get_ring_leaderboard(current_user: Optional[dict] = Depends(get_current_user)):
    """Get the top 10 ring collectors. Public endpoint."""
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT u.display_name, u.username, r.rings_collected
                FROM april_fools_rings r
                JOIN users u ON u.id = r.user_id
                ORDER BY r.rings_collected DESC
                LIMIT 10
                """
            )
            rows = await acur.fetchall()

            viewer = None
            if current_user:
                await acur.execute(
                    """
                    SELECT rings_collected
                    FROM april_fools_rings
                    WHERE user_id = %s
                    """
                    ,
                    (current_user["user_id"],)
                )
                viewer_row = await acur.fetchone()
                viewer_rings = viewer_row[0] if viewer_row else 0

                viewer_rank = None
                if viewer_rings > 0:
                    await acur.execute(
                        """
                        SELECT COUNT(*) + 1
                        FROM april_fools_rings
                        WHERE rings_collected > %s
                        """
                        ,
                        (viewer_rings,)
                    )
                    rank_row = await acur.fetchone()
                    viewer_rank = rank_row[0] if rank_row else None

                top_ten_cutoff = rows[-1][2] if len(rows) == 10 else None
                points_to_top_ten = 0
                if top_ten_cutoff is not None and (viewer_rank is None or viewer_rank > 10):
                    points_to_top_ten = max(0, top_ten_cutoff + 1 - viewer_rings)

                viewer = {
                    "display_name": current_user.get("display_name") or current_user.get("username") or "You",
                    "rings": viewer_rings,
                    "rank": viewer_rank,
                    "points_to_top_ten": points_to_top_ten,
                    "top_ten_cutoff": top_ten_cutoff,
                }

    leaderboard = []
    for i, row in enumerate(rows):
        leaderboard.append({
            "rank": i + 1,
            "display_name": row[0] or row[1] or "Anonymous",
            "rings": row[2],
        })

    return {"leaderboard": leaderboard, "viewer": viewer}

