from __future__ import annotations

import asyncio
import logging
import json
import re
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel

from video_core import (
    VIDEO_ASPECTS,
    VIDEO_MAX_FRAME_BYTES,
    VIDEO_PRICES,
    VIDEO_MAX_REFERENCE_IMAGES, VIDEO_MAX_REFERENCE_VIDEOS,
    VIDEO_MAX_REFERENCE_VIDEO_BYTES, VIDEO_MAX_REFERENCE_TOTAL_BYTES,
    VIDEO_REFERENCE_MIN_SECONDS, VIDEO_REFERENCE_MAX_SECONDS,
    validate_generation_inputs,
    VideoInputError,
    aspect_settings,
    effective_service_state,
    normalize_prompt,
    normalize_video_frame,
    price_for_duration,
    video_frame_needs_watermark,
    watermark_video_frame,
)

from video_references import normalize_reference_video
from video_pricing import VIDEO_PRICING_VERSION, video_price_quote

load_dotenv()


logger = logging.getLogger("mobians.video")

VIDEO_FEATURE_ENABLED = os.environ.get("VIDEO_GENERATION_ENABLED", "0").strip() == "1"
VIDEO_HEARTBEAT_TIMEOUT_SECONDS = max(
    15.0, float(os.environ.get("VIDEO_HEARTBEAT_TIMEOUT_SECONDS", "45"))
)
VIDEO_MEDIA_TOKEN_TTL_SECONDS = max(
    60, int(os.environ.get("VIDEO_MEDIA_TOKEN_TTL_SECONDS", "600"))
)
VIDEO_RESULT_RETENTION_HOURS = 24
VIDEO_ACTIVE_JOB_LIMIT = 3
VIDEO_MAX_SEED = 2**63 - 1
VIDEO_IS_DEV_ENV = (os.environ.get("VIDEO_JOB_ENV") or "production").strip().lower() == "development"


class VideoServiceUpdate(BaseModel):
    desired_state: str
    maintenance_message: Optional[str] = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _heartbeat_age_seconds(value: Optional[datetime]) -> Optional[float]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0.0, (_utcnow() - value).total_seconds())


async def _fetch_service_state(pool: Any) -> dict[str, Any]:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM video_service_state WHERE id = 1")
            row = await cur.fetchone()
    if not row:
        raise RuntimeError("video_service_state row is missing")
    return dict(row)


def _public_service_payload(row: dict[str, Any], *, feature_enabled: bool) -> dict[str, Any]:
    heartbeat_age = _heartbeat_age_seconds(row.get("worker_heartbeat_at"))
    effective_state = effective_service_state(
        feature_enabled=feature_enabled,
        desired_state=str(row.get("desired_state") or "maintenance"),
        worker_status=row.get("worker_status"),
        heartbeat_age_seconds=heartbeat_age,
        heartbeat_timeout_seconds=VIDEO_HEARTBEAT_TIMEOUT_SECONDS,
    )
    default_messages = {
        "disabled": "Video generation is not enabled yet.",
        "offline": "Video generation is temporarily offline.",
        "low_storage": "Video generation is paused while server storage is cleared.",
        "draining": "Video generation is entering maintenance after queued jobs finish.",
        "maintenance": row.get("maintenance_message") or "Video generation is under maintenance.",
    }
    return {
        "feature_enabled": feature_enabled,
        "desired_state": row.get("desired_state") or "maintenance",
        "effective_state": effective_state,
        "accepting_jobs": effective_state == "available",
        "message": default_messages.get(effective_state, "Video generation is available."),
        "maintenance_message": row.get("maintenance_message"),
        "worker_status": row.get("worker_status") or "offline",
        "worker_message": row.get("worker_message"),
        "worker_heartbeat_at": _iso(row.get("worker_heartbeat_at")),
        "heartbeat_age_seconds": round(heartbeat_age, 1) if heartbeat_age is not None else None,
        "disk_free_bytes": row.get("disk_free_bytes"),
        "disk_total_bytes": row.get("disk_total_bytes"),
        "current_job_id": str(row["current_job_id"]) if row.get("current_job_id") else None,
    }


def _job_payload(row: dict[str, Any], queue_positions: Optional[dict[str, int]] = None) -> dict[str, Any]:
    job_id = str(row["id"])
    status = str(row["status"])
    queue_position = None
    if status == "pending" and queue_positions is not None:
        queue_position = queue_positions.get(job_id)
    elif status == "processing":
        queue_position = 0
    return {
        "id": job_id,
        "status": status,
        "created_at": _iso(row.get("create_date")),
        "updated_at": _iso(row.get("updated_at")),
        "started_at": _iso(row.get("started_at")),
        "completed_at": _iso(row.get("completed_at")),
        "expires_at": _iso(row.get("expires_at")),
        "prompt": row.get("prompt") or "",
        "audio_prompt": row.get("audio_prompt"),
        "disable_sound": bool(row.get("disable_sound")),
        "output_format": row.get("output_format") or "video",
        "duration_seconds": row.get("duration_seconds"),
        "aspect_ratio": row.get("aspect_ratio"),
        "width": row.get("width"),
        "height": row.get("height"),
        "seed": row.get("seed"),
        "progress": row.get("progress") or 0,
        "queue_position": queue_position,
        "credit_cost": row.get("credit_cost"),
        "refunded": bool(row.get("refunded")),
        "error_message": row.get("error_message"),
        "has_last_frame": bool(row.get("last_frame_thumbnail")),
        "generation_mode": row.get("generation_mode") or "fl2v",
        "reference_image_count": row.get("reference_image_count") or 0,
        "reference_video_count": row.get("reference_video_count") or 0,
        "media_ready": status == "completed" and bool(row.get("output_filename")),
    }


async def _queue_positions(pool: Any) -> dict[str, int]:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, ROW_NUMBER() OVER (ORDER BY create_date, id) AS queue_position
                FROM video_generation_queue
                WHERE status = 'pending'
                ORDER BY create_date, id
                """
            )
            rows = await cur.fetchall()
    return {str(row["id"]): int(row["queue_position"]) for row in rows}


async def _owned_job(pool: Any, job_id: str, user_id: str) -> dict[str, Any]:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM video_generation_queue WHERE id = %s AND user_id = %s",
                (job_id, user_id),
            )
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Video job not found.")
    return dict(row)


def _media_token(secret: str, *, job_id: str, user_id: str) -> str:
    now = _utcnow()
    return jwt.encode(
        {
            "sub": user_id,
            "job_id": job_id,
            "scope": "video_media",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=VIDEO_MEDIA_TOKEN_TTL_SECONDS)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )


def _decode_media_token(secret: str, token: str, job_id: str) -> str:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Video access token is invalid or expired.") from exc
    if payload.get("scope") != "video_media" or str(payload.get("job_id")) != str(job_id):
        raise HTTPException(status_code=403, detail="Video access token does not match this job.")
    user_id = str(payload.get("sub") or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Video access token is invalid.")
    return user_id


def register_video_routes(
    app: Any,
    *,
    pool_getter: Callable[[], Any],
    session_getter: Callable[[], Any],
    require_auth_dep: Callable[..., Awaitable[dict]],
    get_current_user_dep: Callable[..., Awaitable[Optional[dict]]],
    require_admin_dep: Callable[..., Awaitable[dict]],
    media_token_secret: str,
) -> None:
    @app.get("/video/config")
    async def get_video_config():
        service: dict[str, Any]
        try:
            row = await _fetch_service_state(pool_getter())
            service = _public_service_payload(row, feature_enabled=VIDEO_FEATURE_ENABLED)
        except Exception as exc:
            logger.warning("Video config unavailable: %s", exc)
            service = {
                "feature_enabled": VIDEO_FEATURE_ENABLED,
                "desired_state": "maintenance",
                "effective_state": "offline" if VIDEO_FEATURE_ENABLED else "disabled",
                "accepting_jobs": False,
                "message": "Video generation is temporarily unavailable while setup is completed.",
                "worker_status": "offline",
            }
        return {
            "service": service,
            "prices": {str(key): value for key, value in VIDEO_PRICES.items()},
            "pricing_version": VIDEO_PRICING_VERSION,
            "aspects": VIDEO_ASPECTS,
            "durations": sorted(VIDEO_PRICES),
            "active_job_limit": VIDEO_ACTIVE_JOB_LIMIT,
            "retention_hours": VIDEO_RESULT_RETENTION_HOURS,
            "max_frame_bytes": VIDEO_MAX_FRAME_BYTES,
            "accepted_frame_types": ["image/jpeg", "image/png", "image/webp"],
            "generation_modes": ["fl2v", "ref2v"],
            "max_reference_images": VIDEO_MAX_REFERENCE_IMAGES,
            "max_reference_videos": VIDEO_MAX_REFERENCE_VIDEOS,
            "max_reference_video_bytes": VIDEO_MAX_REFERENCE_VIDEO_BYTES,
            "max_reference_total_bytes": VIDEO_MAX_REFERENCE_TOTAL_BYTES,
            "reference_video_min_seconds": VIDEO_REFERENCE_MIN_SECONDS,
            "reference_video_max_seconds": VIDEO_REFERENCE_MAX_SECONDS,
            "accepted_reference_video_types": ["video/mp4", "video/quicktime", "video/webm"],
        }

    @app.post("/video/quote")
    async def quote_video_job(
        duration_seconds: int = Form(...),
        generation_mode: str = Form("fl2v"),
        reference_image_count: int = Form(0),
        reference_video_seconds: str = Form("[]"),
    ):
        try:
            durations = json.loads(reference_video_seconds)
            if not isinstance(durations, list):
                raise VideoInputError("Reference video durations must be a JSON array.")
            return video_price_quote(duration_seconds, generation_mode, reference_image_count, durations)
        except (VideoInputError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/video/jobs")
    async def submit_video_job(
        first_frame: Optional[UploadFile] = File(None),
        generation_mode: str = Form("fl2v"),
        reference_images: Optional[list[UploadFile]] = File(None),
        reference_videos: Optional[list[UploadFile]] = File(None),
        reference_image_sources: str = Form("[]"),
        reference_video_audio: str = Form("[]"),
        last_frame: Optional[UploadFile] = File(None),
        first_frame_source: str = Form("upload"),
        last_frame_source: str = Form("upload"),
        prompt: str = Form(...),
        audio_prompt: Optional[str] = Form(None),
        disable_sound: bool = Form(False),
        output_format: str = Form("video"),
        duration_seconds: int = Form(...),
        aspect_ratio: str = Form(...),
        seed: Optional[int] = Form(None),
        expected_credit_cost: Optional[int] = Form(None),
        pricing_version: Optional[str] = Form(None),
        user: dict = Depends(require_auth_dep),
    ):
        pool = pool_getter()
        try:
            service_row = await _fetch_service_state(pool)
            service = _public_service_payload(service_row, feature_enabled=VIDEO_FEATURE_ENABLED)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Video generation setup is unavailable.") from exc
        if not service["accepting_jobs"]:
            raise HTTPException(status_code=503, detail=service["message"])

        try:
            image_uploads, video_uploads = reference_images or [], reference_videos or []
            validate_generation_inputs(generation_mode, first_frame is not None, last_frame is not None,
                                       len(image_uploads), len(video_uploads))
            try:
                sources = json.loads(reference_image_sources)
                video_audio = json.loads(reference_video_audio)
            except (ValueError, TypeError) as exc:
                raise VideoInputError("Reference options must be JSON arrays.") from exc
            if not isinstance(sources, list) or not isinstance(video_audio, list):
                raise VideoInputError("Reference options must be JSON arrays.")
            if not sources:
                sources = ["upload"] * len(image_uploads)
            if not video_audio:
                video_audio = [False] * len(video_uploads)
            if len(sources) != len(image_uploads) or len(video_audio) != len(video_uploads):
                raise VideoInputError("Reference options must match the uploaded references.")
            if any(source not in {"upload", "history"} for source in sources if isinstance(source, str)) or any(not isinstance(source, str) for source in sources):
                raise VideoInputError("Reference image sources must be upload or history.")
            if any(not isinstance(value, bool) for value in video_audio):
                raise VideoInputError("Reference video audio choices must be booleans.")
            for kind, count in [("Picture", len(image_uploads)), ("Video", len(video_uploads))]:
                if generation_mode == "ref2v" and any(not 1 <= int(match.group(1)) <= count for match in re.finditer(r"<" + kind + r" (\d+)>", prompt)):
                    raise VideoInputError(f"The prompt refers to a missing {kind.lower()} reference.")
            normalized_output_format = str(output_format or "video").strip().lower()
            if normalized_output_format not in {"video", "gif"}:
                raise VideoInputError("Output format must be video or gif.")
            disable_sound = bool(disable_sound or normalized_output_format == "gif")
            normalized_prompt, normalized_audio = normalize_prompt(
                prompt, None if disable_sound else audio_prompt
            )
            cost = price_for_duration(duration_seconds)
            aspect = aspect_settings(aspect_ratio)
            if seed is None:
                seed = secrets.randbelow(VIDEO_MAX_SEED)
            if seed < 0 or seed > VIDEO_MAX_SEED:
                raise VideoInputError("Seed is outside the supported range.")
            first = last = preview = None
            if first_frame is not None:
                first_bytes = await first_frame.read(VIDEO_MAX_FRAME_BYTES + 1)
                first = await asyncio.to_thread(normalize_video_frame, first_bytes)
                if video_frame_needs_watermark(first_frame_source):
                    first = await asyncio.to_thread(watermark_video_frame, first)
                preview = first
            if last_frame is not None:
                last_bytes = await last_frame.read(VIDEO_MAX_FRAME_BYTES + 1)
                last = await asyncio.to_thread(normalize_video_frame, last_bytes)
                if video_frame_needs_watermark(last_frame_source):
                    last = await asyncio.to_thread(watermark_video_frame, last)
            references = []
            video_durations = []
            total_bytes = 0
            for index, upload in enumerate(image_uploads):
                data = await upload.read(VIDEO_MAX_FRAME_BYTES + 1)
                total_bytes += len(data)
                if total_bytes > VIDEO_MAX_REFERENCE_TOTAL_BYTES:
                    raise VideoInputError("Reference uploads must total 150 MB or less.")
                normalized = await asyncio.to_thread(normalize_video_frame, data)
                if video_frame_needs_watermark(sources[index]):
                    normalized = await asyncio.to_thread(watermark_video_frame, normalized)
                preview = preview or normalized
                references.append(dict(kind="image", position=index, media=normalized.image_bytes,
                                       mime_type=normalized.mime_type, use_audio=False))
            for index, upload in enumerate(video_uploads):
                data = await upload.read(VIDEO_MAX_REFERENCE_VIDEO_BYTES + 1)
                total_bytes += len(data)
                if total_bytes > VIDEO_MAX_REFERENCE_TOTAL_BYTES:
                    raise VideoInputError("Reference uploads must total 150 MB or less.")
                normalized_video = await asyncio.to_thread(normalize_reference_video, data, video_audio[index])
                video_durations.append(normalized_video.duration)
                preview = preview or normalized_video.thumbnail
                references.append(dict(kind="video", position=index, media=normalized_video.media,
                                       mime_type="video/mp4", use_audio=normalized_video.has_audio))
            if sum(len(item["media"]) for item in references) > VIDEO_MAX_REFERENCE_TOTAL_BYTES:
                raise VideoInputError("Normalized references exceed 150 MB. Use fewer or smaller references.")
        except VideoInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Reprice validated files, never client-supplied counts/durations. A
        # stale or missing REF2V quote must be reviewed before any debit/job write.
        quote = video_price_quote(duration_seconds, generation_mode, len(image_uploads), video_durations)
        cost = quote["credit_cost"]
        if ((generation_mode == "ref2v" and
             (expected_credit_cost != cost or pricing_version != VIDEO_PRICING_VERSION))
                or (expected_credit_cost is not None and expected_credit_cost != cost)):
            raise HTTPException(status_code=409, detail={
                "code": "video_price_changed",
                "message": f"The checked price is {cost} credits. Review the updated price and generate again. No credits were charged.",
                "quote": quote,
            })

        job_id = secrets.token_hex(16)
        # token_hex is converted to UUID text so the same id can be linked to credits atomically.
        job_id = f"{job_id[0:8]}-{job_id[8:12]}-{job_id[12:16]}-{job_id[16:20]}-{job_id[20:32]}"
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # Recheck while holding the service row. This closes the window
                # where maintenance/low-storage could begin during image
                # normalization and ensures that no credits are deducted after
                # submissions have been disabled.
                await cur.execute("SELECT * FROM video_service_state WHERE id = 1 FOR SHARE")
                locked_service_row = await cur.fetchone()
                if not locked_service_row:
                    raise HTTPException(status_code=503, detail="Video generation setup is unavailable.")
                locked_service = _public_service_payload(
                    dict(locked_service_row), feature_enabled=VIDEO_FEATURE_ENABLED
                )
                if not locked_service["accepting_jobs"]:
                    raise HTTPException(status_code=503, detail=locked_service["message"])
                await cur.execute("SELECT credits FROM users WHERE id = %s FOR UPDATE", (user["user_id"],))
                user_row = await cur.fetchone()
                if not user_row:
                    raise HTTPException(status_code=404, detail="User account not found.")
                await cur.execute(
                    """
                    SELECT COUNT(*) AS active_count
                    FROM video_generation_queue
                    WHERE user_id = %s AND status IN ('pending', 'processing')
                    """,
                    (user["user_id"],),
                )
                active_count = int((await cur.fetchone())["active_count"])
                if active_count >= VIDEO_ACTIVE_JOB_LIMIT:
                    raise HTTPException(
                        status_code=409,
                        detail="You already have three unfinished video jobs. Wait for one to finish or cancel a pending job.",
                    )
                current_credits = int(user_row["credits"])
                if current_credits < cost:
                    raise HTTPException(
                        status_code=402,
                        detail=f"Insufficient credits. You need {cost} credits but have {current_credits}.",
                    )
                new_balance = current_credits - cost
                await cur.execute("UPDATE users SET credits = %s WHERE id = %s", (new_balance, user["user_id"]))
                await cur.execute(
                    """
                    INSERT INTO video_generation_queue (
                        id, user_id, prompt, audio_prompt, disable_sound, output_format,
                        duration_seconds, aspect_ratio,
                        width, height, seed, first_frame, first_frame_mime,
                        first_frame_thumbnail, first_thumbnail_mime, last_frame,
                        last_frame_mime, last_frame_thumbnail, last_thumbnail_mime,
                        credit_cost, is_dev_job, generation_mode, reference_image_count, reference_video_count
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        job_id,
                        user["user_id"],
                        normalized_prompt,
                        normalized_audio,
                        disable_sound,
                        normalized_output_format,
                        int(duration_seconds),
                        aspect_ratio,
                        aspect["width"],
                        aspect["height"],
                        int(seed),
                        first.image_bytes if first else None,
                        first.mime_type if first else None,
                        preview.thumbnail_bytes if preview else None,
                        preview.thumbnail_mime_type if preview else None,
                        last.image_bytes if last else None,
                        last.mime_type if last else None,
                        last.thumbnail_bytes if last else None,
                        last.thumbnail_mime_type if last else None,
                        cost,
                        VIDEO_IS_DEV_ENV,
                        generation_mode, len(image_uploads), len(video_uploads),
                    ),
                )
                for reference in references:
                    await cur.execute(
                        "INSERT INTO video_generation_references (job_id, kind, position, media, mime_type, use_audio) VALUES (%s, %s, %s, %s, %s, %s)",
                        (job_id, reference["kind"], reference["position"], reference["media"], reference["mime_type"], reference["use_audio"]),
                    )
                await cur.execute(
                    """
                    INSERT INTO credit_transactions (
                        user_id, amount, balance_after, transaction_type, job_id, description
                    ) VALUES (%s, %s, %s, 'video_generation', %s, %s)
                    """,
                    (
                        user["user_id"],
                        -cost,
                        new_balance,
                        job_id,
                        f"MiniMax H3 video generation - {duration_seconds} seconds",
                    ),
                )
            await conn.commit()

        positions = await _queue_positions(pool)
        row = await _owned_job(pool, job_id, user["user_id"])
        return {
            "job": _job_payload(row, positions),
            "credits_used": cost,
            "credits_remaining": new_balance,
        }

    @app.get("/video/jobs")
    async def list_video_jobs(user: dict = Depends(require_auth_dep)):
        pool = pool_getter()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT *
                    FROM video_generation_queue
                    WHERE user_id = %s
                      AND (
                            status IN ('pending', 'processing')
                            OR (status = 'completed' AND expires_at > NOW())
                            OR (status IN ('failed', 'cancelled') AND updated_at > NOW() - INTERVAL '24 hours')
                          )
                    ORDER BY create_date DESC
                    LIMIT 50
                    """,
                    (user["user_id"],),
                )
                rows = [dict(row) for row in await cur.fetchall()]
        positions = await _queue_positions(pool)
        return {"jobs": [_job_payload(row, positions) for row in rows]}

    @app.get("/video/jobs/{job_id}")
    async def get_video_job(job_id: str, user: dict = Depends(require_auth_dep)):
        pool = pool_getter()
        row = await _owned_job(pool, job_id, user["user_id"])
        return {"job": _job_payload(row, await _queue_positions(pool))}

    @app.delete("/video/jobs/{job_id}")
    async def cancel_video_job(job_id: str, user: dict = Depends(require_auth_dep)):
        pool = pool_getter()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM video_generation_queue WHERE id = %s AND user_id = %s FOR UPDATE",
                    (job_id, user["user_id"]),
                )
                row = await cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Video job not found.")
                if row["status"] != "pending":
                    raise HTTPException(status_code=409, detail="Only pending video jobs can be cancelled.")
                await cur.execute(
                    """
                    UPDATE video_generation_queue
                    SET status = 'cancelled', updated_at = NOW(),
                        error_message = 'Cancelled by user',
                        first_frame = NULL, last_frame = NULL,
                        expires_at = NOW() + INTERVAL '24 hours'
                    WHERE id = %s
                    """,
                    (job_id,),
                )
                await cur.execute(
                    "SELECT * FROM safe_refund_video_credits(%s, %s)",
                    (job_id, "Pending video job cancelled by user"),
                )
                refund = await cur.fetchone()
                await cur.execute("UPDATE video_generation_references SET media = NULL WHERE job_id = %s", (job_id,))
            await conn.commit()
        return {
            "status": "cancelled",
            "credits_refunded": int(refund["credits_refunded"] if refund else 0),
            "credits_remaining": int(refund["new_balance"] if refund else 0),
        }

    @app.get("/video/jobs/{job_id}/frames/{frame_name}/thumbnail")
    async def get_video_frame_thumbnail(
        job_id: str,
        frame_name: str,
        user: dict = Depends(require_auth_dep),
    ):
        if frame_name not in {"first", "last"}:
            raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
        row = await _owned_job(pool_getter(), job_id, user["user_id"])
        data = row.get(f"{frame_name}_frame_thumbnail")
        mime = row.get(f"{frame_name}_thumbnail_mime")
        if not data:
            raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
        return Response(content=bytes(data), media_type=mime or "image/webp", headers={"Cache-Control": "private, max-age=300"})

    @app.post("/video/jobs/{job_id}/media-token")
    async def create_video_media_token(job_id: str, user: dict = Depends(require_auth_dep)):
        await _owned_job(pool_getter(), job_id, user["user_id"])
        return {
            "access_token": _media_token(media_token_secret, job_id=job_id, user_id=user["user_id"]),
            "expires_in": VIDEO_MEDIA_TOKEN_TTL_SECONDS,
        }

    async def stream_video_media(
        job_id: str,
        request: Request,
        *,
        download: bool,
        access_token: Optional[str],
        user: Optional[dict],
    ):
        user_id = user.get("user_id") if user else None
        if not user_id and access_token:
            user_id = _decode_media_token(media_token_secret, access_token, job_id)
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required.")
        row = await _owned_job(pool_getter(), job_id, str(user_id))
        if row["status"] == "expired" or (row.get("expires_at") and row["expires_at"] <= _utcnow()):
            raise HTTPException(status_code=410, detail="This video has expired.")
        if row["status"] != "completed" or not row.get("output_filename"):
            raise HTTPException(status_code=409, detail="Video output is not ready.")

        try:
            service = _public_service_payload(
                await _fetch_service_state(pool_getter()), feature_enabled=VIDEO_FEATURE_ENABLED
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Video storage is temporarily unavailable.") from exc
        if service["effective_state"] in {"offline", "disabled"}:
            raise HTTPException(status_code=503, detail="The video server is temporarily offline.")

        session = session_getter()
        if session is None:
            raise HTTPException(status_code=503, detail="Video storage connection is unavailable.")
        comfy_url = (os.environ.get("COMFYUI_BASE_URL") or "http://127.0.0.1:8188").rstrip("/")
        headers: dict[str, str] = {}
        if request.headers.get("range"):
            headers["Range"] = request.headers["range"]
        try:
            upstream = await session.get(
                f"{comfy_url}/view",
                params={
                    "filename": row["output_filename"],
                    "subfolder": row.get("output_subfolder") or "",
                    "type": row.get("output_type") or "output",
                },
                headers=headers,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="The video server is temporarily offline.") from exc
        if upstream.status not in {200, 206}:
            status = upstream.status
            upstream.release()
            if status == 404:
                raise HTTPException(status_code=404, detail="Video file is unavailable.")
            raise HTTPException(status_code=503, detail="Video storage could not serve this file.")

        response_headers: dict[str, str] = {"Accept-Ranges": "bytes", "Cache-Control": "private, no-store"}
        for header_name in ("Content-Length", "Content-Range"):
            if upstream.headers.get(header_name):
                response_headers[header_name] = upstream.headers[header_name]
        if download:
            safe_name = str(row["output_filename"]).replace('"', "").replace("\r", "").replace("\n", "")
            response_headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

        async def body_iterator():
            try:
                async for chunk in upstream.content.iter_chunked(256 * 1024):
                    yield chunk
            finally:
                upstream.release()

        return StreamingResponse(
            body_iterator(),
            status_code=upstream.status,
            media_type=row.get("output_mime") or upstream.headers.get("Content-Type") or "video/mp4",
            headers=response_headers,
        )

    @app.get("/video/jobs/{job_id}/content")
    async def get_video_content(
        job_id: str,
        request: Request,
        access_token: Optional[str] = Query(None),
        user: Optional[dict] = Depends(get_current_user_dep),
    ):
        return await stream_video_media(
            job_id, request, download=False, access_token=access_token, user=user
        )

    @app.get("/video/jobs/{job_id}/download")
    async def download_video_content(
        job_id: str,
        request: Request,
        access_token: Optional[str] = Query(None),
        user: Optional[dict] = Depends(get_current_user_dep),
    ):
        return await stream_video_media(
            job_id, request, download=True, access_token=access_token, user=user
        )

    @app.get("/admin/video-service")
    async def admin_get_video_service(user: dict = Depends(require_admin_dep)):
        pool = pool_getter()
        row = await _fetch_service_state(pool)
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'pending') AS pending_count,
                        COUNT(*) FILTER (WHERE status = 'processing') AS processing_count,
                        COUNT(*) FILTER (WHERE status = 'completed' AND expires_at > NOW()) AS retained_count
                    FROM video_generation_queue
                    """
                )
                counts = dict(await cur.fetchone())
        return {"service": _public_service_payload(row, feature_enabled=VIDEO_FEATURE_ENABLED), "counts": counts}

    @app.put("/admin/video-service")
    async def admin_update_video_service(
        update: VideoServiceUpdate,
        user: dict = Depends(require_admin_dep),
    ):
        desired_state = str(update.desired_state or "").strip().lower()
        if desired_state not in {"available", "draining", "maintenance"}:
            raise HTTPException(status_code=400, detail="Invalid video service state.")
        message = (update.maintenance_message or "Video generation is temporarily under maintenance.").strip()
        if len(message) > 500:
            raise HTTPException(status_code=400, detail="Maintenance message must be 500 characters or fewer.")
        pool = pool_getter()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                if desired_state == "draining":
                    await cur.execute(
                        "SELECT COUNT(*) AS count FROM video_generation_queue WHERE status IN ('pending', 'processing')"
                    )
                    if int((await cur.fetchone())["count"]) == 0:
                        desired_state = "maintenance"
                await cur.execute(
                    """
                    UPDATE video_service_state
                    SET desired_state = %s, maintenance_message = %s, updated_at = NOW()
                    WHERE id = 1
                    RETURNING *
                    """,
                    (desired_state, message),
                )
                row = dict(await cur.fetchone())
            await conn.commit()
        return {"service": _public_service_payload(row, feature_enabled=VIDEO_FEATURE_ENABLED)}


async def video_completion_notifier_task(
    pool_getter: Callable[[], Any],
    send_push: Callable[[str, dict], Awaitable[int]],
    public_site_url: str,
) -> None:
    await asyncio.sleep(20)
    while True:
        try:
            pool = pool_getter()
            if pool is None:
                await asyncio.sleep(5)
                continue
            async with pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        """
                        UPDATE video_generation_queue
                        SET notified_at = NOW(), updated_at = NOW()
                        WHERE id IN (
                            SELECT id FROM video_generation_queue
                            WHERE notified_at IS NULL
                              AND status IN ('completed', 'failed')
                              AND completed_at > NOW() - INTERVAL '30 minutes'
                            ORDER BY completed_at
                            LIMIT 25
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING user_id, status, credit_cost, refunded
                        """
                    )
                    claimed = [dict(row) for row in await cur.fetchall()]
                await conn.commit()
            for row in claimed:
                completed = row["status"] == "completed"
                title = "Your video is ready!" if completed else "Your video generation failed"
                if completed:
                    body = "Tap to watch and download it within 24 hours."
                elif row.get("refunded"):
                    body = f"Something went wrong. {row['credit_cost']} credits were refunded."
                else:
                    body = "Something went wrong. Please try again."
                await send_push(
                    str(row["user_id"]),
                    {
                        "notification": {
                            "title": title,
                            "body": body,
                            "vibrate": [100, 50, 100],
                            "data": {"url": f"{public_site_url.rstrip('/')}/video"},
                        }
                    },
                )
        except Exception as exc:
            logger.debug("Video notifier waiting for setup: %s", exc)
        await asyncio.sleep(5)
