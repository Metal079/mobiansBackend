import os
import io
import base64
import sys
import asyncio
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime, timedelta
import json
import re
import time
import math
import secrets
import uuid

# Fix for Windows - psycopg async requires SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, UploadFile, File, Query
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

# Define your connection parameters for PostgreSQL
DSN = f"host={DBHOST} dbname='{DBNAME}' user={DBUSER} password={DBPASS}"

VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = os.environ.get("VAPID_CLAIMS")
subscriptions: Dict[str, dict] = {}

# Credit costs by model type
CREDIT_COSTS = {
    "SD 1.5": 10,      # sonicDiffusionV4
    "Pony": 15,        # autismMix (SDXL-based)
    "Illustrious": 15,  # novaFurryXL_ilV140 (SDXL-based), novaMobianXL_v10, novaMobianXL_v20
    "Anima": 20        # Anima-preview2
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
    "Anima-preview2": "Anima",
}

DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "novaMobianXL_v20")
LORA_SUGGESTION_LIMIT = 5
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
    base_type = MODEL_BASE_TYPES.get(model, "SD 1.5")
    base_cost = CREDIT_COSTS.get(base_type, CREDIT_COSTS.get("SD 1.5", 0))
    lora_count = len(loras) if isinstance(loras, list) else 0
    per_lora_cost = LORA_CREDIT_COSTS.get(base_type, 0)
    return base_cost + (lora_count * per_lora_cost)

def get_upscale_credit_cost(model: str, loras: Optional[List[Dict[str, Any]]] = None) -> int:
    """Get the credit cost for an upscale job (base model cost * multiplier + LoRA costs * 3)."""
    base_type = MODEL_BASE_TYPES.get(model, "SD 1.5")
    base_cost = CREDIT_COSTS.get(base_type, CREDIT_COSTS.get("SD 1.5", 0))
    lora_count = len(loras) if isinstance(loras, list) else 0
    per_lora_cost = LORA_CREDIT_COSTS.get(base_type, 0)
    lora_total = lora_count * per_lora_cost * UPSCALE_CREDIT_MULTIPLIER
    return (base_cost * UPSCALE_CREDIT_MULTIPLIER) + lora_total


def get_hires_credit_cost(model: str, loras: Optional[List[Dict[str, Any]]] = None) -> int:
    """Get the credit cost for a hi-res (generate+upscale) job (base model cost * 4 + LoRA costs * 4)."""
    base_type = MODEL_BASE_TYPES.get(model, "SD 1.5")
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
    # New fields for credit system
    queue_type: Optional[str] = "free"  # "free" or "priority"


class ImageRequestModel(JobData):
    image: Optional[str] = None
    fast_pass_enabled: Optional[bool] = False
    user_id: Optional[str] = None
    credit_cost: Optional[int] = 0


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

    # Filter out prompts
    job_data.prompt, job_data.negative_prompt = await promptFilter(job_data)
    job_data.negative_prompt = await fortify_default_negative(job_data.negative_prompt)

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
                    id, status, assigned_gpu, prompt, image, image_UUID, mask_image,
                    color_inpaint, control_image, scheduler, steps, negative_prompt,
                    width, height, guidance_scale, seed, batch_size, strength,
                    job_type, model, fast_pass_code, rating, enable_upscale, fast_pass_enabled, 
                    is_dev_job, loras, lossy_images, user_id, queue_type, credit_cost
                ) VALUES (
                    gen_random_uuid(), 'pending', NULL, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id;
            """,
                (
                    image_request_data.prompt,
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

    response_data = {"job_id": str(job_id[0]), "queue_type": queue_type}
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

    if not show_nsfw:
        # aiohttp/yarl reject bool query param values; send as string.
        params["nsfw"] = "false"

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

    if not show_nsfw:
        # aiohttp/yarl reject bool query param values; send as string.
        params["nsfw"] = "false"

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
    requestor: str
    is_nsfw: bool
    is_minor: bool
    preview_image: str
    base_model: Optional[str] = None

@app.post("/add_lora_suggestion/")
async def add_lora_suggestion(lora_data: addLoraSuggestion):
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
    if lora_data.requestor:
        async with db_pool.connection() as aconn:
            async with aconn.cursor() as acur:
                await acur.execute(
                    """
                    SELECT COUNT(*)
                    FROM lora_suggestions
                    WHERE requestor = %s
                      AND status = 'pending'
                    """,
                    (lora_data.requestor,)
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
                        lora_data.requestor,
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
                    SELECT status
                    FROM lora_suggestions
                    WHERE version_id = %s
                    """,
                    (lora_data.lora_version_id,),
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
        # Allow re-queueing when the previous attempt already finished processing
        immutable_statuses = {"pending", "approved", "downloading", "rejected"}
        if existing_status in immutable_statuses:
            detail = "A suggestion for this LoRA is already pending approval. Please be patient as we review it."
            if existing_status == "approved":
                detail = "This LoRA has already been approved and is queued for download."
            elif existing_status == "downloading":
                detail = "This LoRA is currently downloading. Please wait for it to finish."
            elif existing_status == "rejected":
                detail = "This LoRA has been reviewed and rejected. It cannot be re-submitted."
            return JSONResponse(content={"status": "error", "detail": detail}, status_code=400)

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
                        lora_data.requestor,
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


async def insert_image_hashes(image_hashes, metadata, job_data):
    logging.info("Inserting image hashes")

    lora_text = ""
    if metadata["loras"]:
        for lora in metadata["loras"]:
            lora_text += f"{lora['name']} - {lora['version']} - strength: {lora['strength']}\n"

    insert_query = """
        INSERT INTO hashes (hash, prompt, negative_prompt, seed, cfg, model, created_date, loras, job_id, finished_images_index)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = [
        (
            image_hashes[i],
            metadata["prompt"],
            metadata["negative_prompt"],
            metadata["seed"],
            metadata["guidance_scale"],
            metadata["model"],
            datetime.now(),
            lora_text,
            job_data.job_id,
            i+1,
        )
        for i in range(4)
    ]

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Use executemany to insert multiple records
            await acur.executemany(insert_query, values)
            await aconn.commit()  # Commit the transaction


async def twos_complement(hexstr, bits):
    value = int(hexstr, 16)  # convert hexadecimal to integer

    # convert from unsigned number to signed number with "bits" bits
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


async def process_images_and_store_hashes(image_results, metadata, job_data):
    image_hashes = []
    for i in range(4):
        image = decode_base64_to_image(image_results[i])
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


@app.post("/get_job/")
async def get_job(job_data: GetJobData, background_tasks: BackgroundTasks):
    metadata = {}
    error_message = None
    refund_info = None  # Will be populated if a refund is issued

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            # Query both view (for queue_position) and base table (for credit/refund info)
            await acur.execute(
                """
                SELECT v.status, v.queue_position, v.finished_images, v.prompt, v.negative_prompt, 
                       v.seed, v.guidance_scale, v.job_type, v.model, v.error_message, v.loras, v.lossy_images,
                       g.user_id, g.credit_cost, COALESCE(g.refunded, FALSE) as refunded,
                       g.control_image
                FROM vw_generation_queue v
                JOIN generation_queue g ON v.id = g.id
                WHERE v.id = %s
            """,
                (job_data.job_id,),
            )
            result = await acur.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    (
        job_status,
        queue_position,
        finished_images,
        metadata["prompt"],
        metadata["negative_prompt"],
        metadata["seed"],
        metadata["guidance_scale"],
        metadata["job_type"],
        metadata["model"],
        error_message,
        metadata['loras'],
        metadata['lossy_images'],
        job_user_id,
        job_credit_cost,
        job_refunded,
        raw_control_image,
    ) = result

    # Parse regional prompting from control_image if present
    metadata['regional_prompting'] = None
    if raw_control_image and isinstance(raw_control_image, str) and raw_control_image.startswith('__regional_prompting__:'):
        try:
            metadata['regional_prompting'] = json.loads(raw_control_image[len('__regional_prompting__:'):])
        except (json.JSONDecodeError, TypeError):
            pass

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
                content={"status": "completed", "result": watermarked_image_base64}
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
async def get_job_status(job_data: GetJobData):
    """Lightweight status-only endpoint for polling. Returns no image data."""
    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT v.status, v.queue_position, v.error_message,
                       g.user_id, g.credit_cost, COALESCE(g.refunded, FALSE) as refunded
                FROM vw_generation_queue v
                JOIN generation_queue g ON v.id = g.id
                WHERE v.id = %s
                """,
                (job_data.job_id,),
            )
            result = await acur.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status, queue_position, error_message, job_user_id, job_credit_cost, job_refunded = result

    if job_status == "completed":
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

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                SELECT v.status, v.finished_images, v.prompt, v.negative_prompt,
                       v.seed, v.guidance_scale, v.job_type, v.model, v.loras, v.lossy_images,
                       g.control_image
                FROM vw_generation_queue v
                JOIN generation_queue g ON v.id = g.id
                WHERE v.id = %s
                """,
                (job_id,),
            )
            result = await acur.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    (
        job_status, finished_images, prompt, negative_prompt,
        seed, guidance_scale, job_type, model, loras, lossy_images,
        raw_control_image,
    ) = result

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
        "regional_prompting": None,
    }

    if raw_control_image and isinstance(raw_control_image, str) and raw_control_image.startswith('__regional_prompting__:'):
        try:
            metadata['regional_prompting'] = json.loads(raw_control_image[len('__regional_prompting__:'):])
        except (json.JSONDecodeError, TypeError):
            pass

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
                SELECT * FROM lora_metadata
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
            for item in result:
                item.pop('image_blob', None)
    
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


@app.post("/subscribe")
async def subscribe(subscription: Subscription):
    user_id = subscription.userId
    subscriptions[user_id] = subscription.dict()
    return {"status": "subscribed"}


@app.get("/send_notification/{user_id}")
async def send_notification(user_id: str):  # Change the type to str
    # Retrieve the subscription object for the user
    subscription = subscriptions.get(user_id)
    if not subscription:
        return {"status": "failed", "detail": "User not subscribed"}

    try:
        payload = {
            "notification": {
                "title": "Your image is ready!",
                "body": "Click to view your image.",
                # "icon": "icon.png",
                "vibrate": [100, 50, 100],
                "data": {"url": "https://mobians.ai/"},
            }
        }
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={"sub": "mailto:your_email@example.com"},
        )
    except WebPushException as e:
        print("Failed to send notification:", repr(e))
        return {"status": "failed", "detail": repr(e)}

    return {"status": "sent"}


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


async def resolve_civitai_model_link(version_id: int) -> str:
    """Resolve a CivitAI model page URL from a model version id."""
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
    requestor = user.get("discord_user_id") or user.get("user_id") or user.get("username")
    if not requestor:
        raise HTTPException(status_code=400, detail="No requestor id available")

    status_norm = (status or "pending").strip().lower()

    where_clause = "WHERE requestor = %s"
    params: list = [str(requestor)]
    if status_norm != "all":
        where_clause += " AND status = %s"
        params.append(status_norm)

    async with db_pool.connection() as aconn:
        async with aconn.cursor() as acur:
            await acur.execute(
                """
                  SELECT version_id, name, version, status, requestor, 
                      is_nsfw, is_minor, preview_image, base_model,
                      error_message, last_updated_date
                FROM lora_suggestions
                """ + where_clause + """
                ORDER BY name
                """,
                tuple(params),
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
                SELECT version_id, status
                FROM lora_suggestions
                WHERE status IN ('rejected', 'approved', 'pending', 'downloading')
                """
            )
            rows = await acur.fetchall()

    result: dict = {"rejected": [], "approved": [], "pending": [], "downloading": []}
    for version_id, status in rows:
        key = (status or "").strip().lower()
        if key in result:
            result[key].append(version_id)
    return JSONResponse(content=result)


@app.post("/cancel_lora_suggestion/{suggestion_id}/")
async def cancel_lora_suggestion(suggestion_id: int, user: dict = Depends(require_auth)):
    """Cancel a pending LoRA suggestion for the current user."""
    requestor = user.get("discord_user_id") or user.get("user_id") or user.get("username")
    if not requestor:
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
            if str(row_requestor) != str(requestor):
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
                      error_message, last_updated_date
                FROM lora_suggestions
                """ + where_clause + """
                ORDER BY name
                """,
                params,
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
                    "UPDATE lora_suggestions SET status = 'duplicate' WHERE version_id = %s",
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
                SET status = 'rejected'
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
                SELECT status FROM vw_generation_queue WHERE id = %s;
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
                    user_id, image_uuid, prompt, prompt_summary, negative_prompt,
                    model, seed, cfg, width, height, aspect_ratio,
                    is_favorite, sync_priority, loras, regional_prompting, tags, image_blob, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (user_id, image_uuid) DO UPDATE SET
                    prompt = EXCLUDED.prompt,
                    prompt_summary = EXCLUDED.prompt_summary,
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
                    request.negative_prompt, request.model, request.seed, request.cfg,
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
                SELECT image_uuid, prompt, prompt_summary, negative_prompt, model,
                       seed, cfg, width, height, aspect_ratio, is_favorite,
                       sync_priority, loras, regional_prompting, tags, image_blob, created_at
                FROM user_synced_images
                WHERE user_id = %s
                ORDER BY sync_priority DESC, created_at DESC
                """
            else:
                query = """
                SELECT image_uuid, prompt, prompt_summary, negative_prompt, model,
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
        created_at = row[16] if include_blobs else row[15]
        if include_blobs and row[15]:
            image_blob_b64 = base64.b64encode(row[15]).decode('utf-8')
        
        images.append({
            "image_uuid": row[0],
            "prompt": row[1],
            "prompt_summary": row[2],
            "negative_prompt": row[3],
            "model": row[4],
            "seed": row[5],
            "cfg": float(row[6]) if row[6] else None,
            "width": row[7],
            "height": row[8],
            "aspect_ratio": row[9],
            "is_favorite": row[10],
            "sync_priority": row[11],
            "loras": row[12] if row[12] else [],
            "regional_prompting": row[13] if row[13] else {"enabled": False, "regions": []},
            "tags": row[14] if row[14] else [],
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
async def get_ring_leaderboard():
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

    leaderboard = []
    for i, row in enumerate(rows):
        leaderboard.append({
            "rank": i + 1,
            "display_name": row[0] or row[1] or "Anonymous",
            "rings": row[2],
        })

    return {"leaderboard": leaderboard}

