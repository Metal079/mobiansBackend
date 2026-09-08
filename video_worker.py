"""Dedicated MiniMax H3 queue worker intended to run beside ComfyUI.

The web backend owns authentication, charging, and media authorization. This
single worker owns ComfyUI submission and local-file lifecycle so multiple web
processes can safely serve one FIFO queue without duplicating generations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import psycopg_pool
from dotenv import load_dotenv
from psycopg.rows import dict_row

from video_core import build_minimax_h3_workflow, load_workflow_template


load_dotenv()


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mobians.video_worker")

COMFY_BASE_URL = (os.environ.get("COMFYUI_BASE_URL") or "http://127.0.0.1:8188").rstrip("/")
COMFY_OUTPUT_ROOT = Path(os.environ.get("COMFYUI_OUTPUT_DIR") or r"D:\ComfyUI\output").resolve()
COMFY_INPUT_ROOT = Path(os.environ.get("COMFYUI_INPUT_DIR") or r"D:\ComfyUI\input").resolve()
POLL_SECONDS = max(1.0, float(os.environ.get("VIDEO_WORKER_POLL_SECONDS", "3")))
HEARTBEAT_SECONDS = max(2.0, float(os.environ.get("VIDEO_WORKER_HEARTBEAT_SECONDS", "5")))
GENERATION_TIMEOUT_SECONDS = max(300.0, float(os.environ.get("VIDEO_GENERATION_TIMEOUT_SECONDS", "7200")))
MIN_DISK_BYTES = max(1, int(os.environ.get("VIDEO_MIN_FREE_BYTES", str(20 * 1024**3))))
VIDEO_JOB_ENV = (os.environ.get("VIDEO_JOB_ENV") or "production").strip().lower()
WORKER_LOCK_ID = 486_433_003

EXPECTED_MODELS = {
    "minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors",
    "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_video_vae_int8_convrot.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
}


def _dsn() -> str:
    explicit = os.environ.get("DATABASE_URL") or os.environ.get("VIDEO_DATABASE_URL")
    if explicit:
        return explicit
    return (
        f"host={os.environ.get('DBHOST')} dbname='{os.environ.get('DBNAME')}' "
        f"user={os.environ.get('DBUSER')} password={os.environ.get('DBPASS')} "
        "application_name=mobians_video_worker keepalives=1 keepalives_idle=30 "
        "keepalives_interval=10 keepalives_count=3 "
        "options='-c idle_in_transaction_session_timeout=30000 -c statement_timeout=60000'"
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_local_path(root: Path, subfolder: str | None, filename: str) -> Path:
    candidate = (root / (subfolder or "") / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("ComfyUI returned a path outside the configured storage sandbox") from exc
    return candidate


def _video_output(history: Any) -> dict[str, str] | None:
    candidates: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("filename"):
                candidates.append(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(history)
    video_extensions = {".mp4", ".webm", ".mov", ".mkv"}
    for item in candidates:
        filename = str(item.get("filename") or "")
        if Path(filename).suffix.lower() in video_extensions:
            return {
                "filename": filename,
                "subfolder": str(item.get("subfolder") or ""),
                "type": str(item.get("type") or "output"),
            }
    return None


def _gif_conversion_status() -> tuple[str | None, str]:
    install_command = f'"{sys.executable}" -m pip install "imageio-ffmpeg>=0.6.0"'
    try:
        import imageio_ffmpeg
    except ImportError:
        return None, f"GIF conversion dependency is missing. Run: {install_command}"

    try:
        executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    except Exception as exc:
        return None, f"GIF conversion executable could not be resolved: {exc}"
    if not executable.is_file():
        return None, f"GIF conversion executable was not found at {executable}"
    return str(executable), f"GIF conversion ready at {executable}"


class VideoWorker:
    def __init__(self) -> None:
        self.pool = psycopg_pool.AsyncConnectionPool(
            _dsn(), min_size=1, max_size=3, timeout=10, open=False
        )
        self.lock_connection: Any = None
        self.http: aiohttp.ClientSession | None = None
        self.stop_event = asyncio.Event()
        self.current_job_id: str | None = None
        self.worker_status = "starting"
        self.worker_message = "Starting video worker"
        self.disk_free_bytes: int | None = None
        self.disk_total_bytes: int | None = None
        self.template = load_workflow_template()
        logger.info(
            "Video workflow: steps=%s sampler=%s scheduler=%s lora=%s strength=%s",
            self.template["124"]["inputs"]["steps"],
            self.template["123"]["inputs"]["sampler_name"],
            self.template["124"]["inputs"]["scheduler"],
            self.template["143"]["inputs"]["lora_name"],
            self.template["143"]["inputs"]["strength_model"],
        )
        self.health_valid_until = 0.0

    async def start(self) -> None:
        await self.pool.open()
        self.http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=60), trust_env=True
        )
        if not await self._take_singleton_lock():
            raise RuntimeError("Another MiniMax H3 video worker already holds the database lock")
        await self.cleanup_expired_files()
        await self.reconcile_processing_jobs()
        heartbeat = asyncio.create_task(self.heartbeat_loop())
        cleanup = asyncio.create_task(self.cleanup_loop())
        try:
            await self.work_loop()
        finally:
            heartbeat.cancel()
            cleanup.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            with suppress(asyncio.CancelledError):
                await cleanup
            await self._publish_heartbeat("offline", "Video worker stopped")
            if self.http:
                await self.http.close()
            if self.lock_connection is not None:
                await self.pool.putconn(self.lock_connection)
            await self.pool.close()

    async def _take_singleton_lock(self) -> bool:
        # Session-level advisory locks are connection-scoped, so keep this pool
        # connection checked out for the full worker lifetime.
        self.lock_connection = await self.pool.getconn()
        async with self.lock_connection.cursor() as cur:
            await cur.execute("SELECT pg_try_advisory_lock(%s)", (WORKER_LOCK_ID,))
            row = await cur.fetchone()
        await self.lock_connection.commit()
        if not row or not row[0]:
            await self.pool.putconn(self.lock_connection)
            self.lock_connection = None
            return False
        return True

    def request_stop(self) -> None:
        self.stop_event.set()

    def _storage_status(self) -> tuple[bool, str]:
        try:
            COMFY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(COMFY_OUTPUT_ROOT)
            self.disk_free_bytes = usage.free
            self.disk_total_bytes = usage.total
            minimum = max(MIN_DISK_BYTES, int(usage.total * 0.10))
            if usage.free < minimum:
                return False, f"Low disk space: {usage.free / 1024**3:.1f} GB free; {minimum / 1024**3:.1f} GB required"
            return True, f"{usage.free / 1024**3:.1f} GB available"
        except OSError as exc:
            self.disk_free_bytes = None
            self.disk_total_bytes = None
            return False, f"Output storage unavailable: {exc}"

    async def _publish_heartbeat(self, status: str | None = None, message: str | None = None) -> None:
        if status:
            self.worker_status = status
        if message is not None:
            self.worker_message = message[:500]
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        UPDATE video_service_state
                        SET worker_heartbeat_at = NOW(), worker_status = %s,
                            worker_message = %s, current_job_id = %s,
                            disk_free_bytes = %s, disk_total_bytes = %s, updated_at = NOW()
                        WHERE id = 1
                        """,
                        (
                            self.worker_status,
                            self.worker_message,
                            self.current_job_id,
                            self.disk_free_bytes,
                            self.disk_total_bytes,
                        ),
                    )
                    if self.current_job_id:
                        await cur.execute(
                            """
                            UPDATE video_generation_queue
                            SET worker_heartbeat_at = NOW(), updated_at = NOW()
                            WHERE id = %s AND status = 'processing'
                            """,
                            (self.current_job_id,),
                        )
                await conn.commit()
        except Exception as exc:
            logger.warning("Could not publish worker heartbeat: %s", exc)

    async def heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            storage_ok, storage_message = self._storage_status()
            if not storage_ok and not self.current_job_id:
                await self._publish_heartbeat("low_storage", storage_message)
            else:
                await self._publish_heartbeat(message=storage_message if not self.current_job_id else self.worker_message)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def _desired_state(self) -> str:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT desired_state FROM video_service_state WHERE id = 1")
                row = await cur.fetchone()
        return str(row[0] if row else "maintenance")

    async def comfy_health(self, *, force: bool = False) -> tuple[bool, str]:
        assert self.http is not None
        gif_executable, gif_message = _gif_conversion_status()
        if not gif_executable:
            return False, gif_message
        now = asyncio.get_running_loop().time()
        if not force and now < self.health_valid_until:
            return True, "ComfyUI nodes, models, and GIF conversion validated"
        try:
            async with self.http.get(f"{COMFY_BASE_URL}/system_stats") as response:
                if response.status != 200:
                    return False, f"ComfyUI system_stats returned HTTP {response.status}"
                await response.read()
            async with self.http.get(f"{COMFY_BASE_URL}/object_info") as response:
                if response.status != 200:
                    return False, f"ComfyUI object_info returned HTTP {response.status}"
                object_info = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return False, f"ComfyUI offline: {exc}"

        required_nodes = set()
        for mode in ("fl2v", "ref2v"):
            sample = build_minimax_h3_workflow(
                generation_mode=mode, first_input_name="first.png" if mode == "fl2v" else None,
                reference_images=["reference.png"] if mode == "ref2v" else None,
                reference_videos=[{"name": "reference.mp4", "use_audio": True}] if mode == "ref2v" else None,
                prompt="Health check", audio_prompt=None, duration_seconds=5,
                aspect_ratio="square", seed=0, job_id="health", template=self.template,
            )
            required_nodes.update(node["class_type"] for node in sample.values())
        missing_nodes = sorted(required_nodes.difference(object_info))
        if missing_nodes:
            return False, "Missing ComfyUI nodes: " + ", ".join(missing_nodes)
        serialized = json.dumps(object_info)
        missing_models = sorted(model for model in EXPECTED_MODELS if model not in serialized)
        if missing_models:
            return False, "Missing workflow models: " + ", ".join(missing_models)
        self.health_valid_until = now + 60
        return True, "ComfyUI nodes, models, and GIF conversion validated"

    async def _claim_job(self) -> dict[str, Any] | None:
        env_filter = ""
        if VIDEO_JOB_ENV == "production":
            env_filter = "AND is_dev_job = FALSE"
        elif VIDEO_JOB_ENV == "development":
            env_filter = "AND is_dev_job = TRUE"
        sql = f"""
            UPDATE video_generation_queue
            SET status = 'processing', started_at = COALESCE(started_at, NOW()),
                worker_heartbeat_at = NOW(), progress = GREATEST(progress, 1), updated_at = NOW()
            WHERE id = (
                SELECT id FROM video_generation_queue
                WHERE status = 'pending' {env_filter}
                ORDER BY create_date, id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
        """
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql)
                row = await cur.fetchone()
            await conn.commit()
        return dict(row) if row else None

    async def _queue_empty(self) -> bool:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT NOT EXISTS (SELECT 1 FROM video_generation_queue WHERE status IN ('pending', 'processing'))"
                )
                row = await cur.fetchone()
        return bool(row and row[0])

    async def _has_processing_jobs(self) -> bool:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM video_generation_queue WHERE status = 'processing')"
                )
                row = await cur.fetchone()
        return bool(row and row[0])

    async def _finish_drain_if_empty(self) -> None:
        if await self._desired_state() == "draining" and await self._queue_empty():
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        UPDATE video_service_state
                        SET desired_state = 'maintenance', updated_at = NOW()
                        WHERE id = 1 AND desired_state = 'draining'
                        """
                    )
                await conn.commit()

    async def work_loop(self) -> None:
        while not self.stop_event.is_set():
            desired_state = await self._desired_state()
            if desired_state == "maintenance":
                await self._publish_heartbeat("idle", "Video service is in maintenance")
                await self._wait_poll()
                continue

            storage_ok, storage_message = self._storage_status()
            if not storage_ok:
                await self._publish_heartbeat("low_storage", storage_message)
                await self._wait_poll()
                continue

            healthy, health_message = await self.comfy_health()
            if not healthy:
                await self._publish_heartbeat("offline", health_message)
                await self._wait_poll()
                continue

            if await self._has_processing_jobs():
                await self.reconcile_processing_jobs(skip_health_check=True)

            job = await self._claim_job()
            if not job:
                await self._finish_drain_if_empty()
                await self._publish_heartbeat("idle", "Ready for video jobs")
                await self._wait_poll()
                continue

            self.current_job_id = str(job["id"])
            await self._publish_heartbeat("processing", "Generating video")
            try:
                await self.process_job(job)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                # The job remains processing through a transient outage. On the
                # next healthy startup/recovery pass, Comfy history is reconciled.
                logger.warning("ComfyUI connection lost for job %s: %s", self.current_job_id, exc)
                await self._publish_heartbeat("offline", f"ComfyUI connection lost: {exc}")
            except Exception as exc:
                logger.exception("Video job %s failed", self.current_job_id)
                await self.fail_job(self.current_job_id, str(exc))
            finally:
                self.current_job_id = None
                await self._finish_drain_if_empty()

    async def _wait_poll(self) -> None:
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=POLL_SECONDS)
        except asyncio.TimeoutError:
            pass

    async def upload_frame(self, job_id: str, frame_name: str, data: bytes, mime_type: str = "image/png") -> str:
        assert self.http is not None
        extension = "mp4" if mime_type == "video/mp4" else "png"
        filename = f"{job_id}_{frame_name}.{extension}"
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename, content_type=mime_type)
        form.add_field("type", "input")
        form.add_field("subfolder", "mobians_video")
        form.add_field("overwrite", "true")
        async with self.http.post(f"{COMFY_BASE_URL}/upload/image", data=form) as response:
            payload = await response.json(content_type=None)
            if response.status not in {200, 201}:
                raise RuntimeError(f"ComfyUI frame upload failed ({response.status}): {payload}")
        name = str(payload.get("name") or Path(filename).name)
        subfolder = str(payload.get("subfolder") or "")
        return f"{subfolder}/{name}".strip("/").replace("\\", "/")

    async def reference_inputs(self, job_id: str) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT * FROM video_generation_references WHERE job_id = %s ORDER BY kind, position", (job_id,))
                return [dict(row) for row in await cur.fetchall()]

    async def process_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["id"])
        mode = job.get("generation_mode") or "fl2v"
        first_name = last_name = None
        reference_images: list[str] = []
        reference_videos: list[dict[str, Any]] = []
        if mode == "fl2v":
            if not job.get("first_frame"):
                raise RuntimeError("First-frame data is missing")
            first_name = await self.upload_frame(job_id, "first", bytes(job["first_frame"]))
            async with self.pool.connection() as conn:
                await conn.execute("UPDATE video_generation_queue SET first_input_name = %s WHERE id = %s", (first_name, job_id))
            if job.get("last_frame"):
                last_name = await self.upload_frame(job_id, "last", bytes(job["last_frame"]))
                async with self.pool.connection() as conn:
                    await conn.execute("UPDATE video_generation_queue SET last_input_name = %s WHERE id = %s", (last_name, job_id))
        else:
            references = await self.reference_inputs(job_id)
            if len(references) != int(job.get("reference_image_count") or 0) + int(job.get("reference_video_count") or 0):
                raise RuntimeError("Reference media is missing")
            for reference in references:
                if not reference.get("media"):
                    raise RuntimeError("Reference media data is missing")
                name = await self.upload_frame(job_id, f"ref_{reference['kind']}_{reference['position']}",
                                               bytes(reference["media"]), reference["mime_type"])
                # Retain source bytes until ComfyUI accepts the prompt, so a
                # restart during upload can recover the job.
                async with self.pool.connection() as conn:
                    await conn.execute("UPDATE video_generation_references SET input_name = %s WHERE job_id = %s AND kind = %s AND position = %s",
                                       (name, job_id, reference["kind"], reference["position"]))
                if reference["kind"] == "image":
                    reference_images.append(name)
                else:
                    reference_videos.append({"name": name, "use_audio": bool(reference["use_audio"])})

        workflow = build_minimax_h3_workflow(
            generation_mode=mode,
            reference_images=reference_images,
            reference_videos=reference_videos,
            first_input_name=first_name,
            last_input_name=last_name,
            prompt=str(job["prompt"]),
            audio_prompt=job.get("audio_prompt"),
            duration_seconds=int(job["duration_seconds"]),
            aspect_ratio=str(job["aspect_ratio"]),
            seed=int(job["seed"]),
            job_id=job_id,
            disable_sound=bool(job.get("disable_sound")),
            template=self.template,
        )
        prompt_id, client_id = await self.submit_prompt(job_id, workflow, first_name, last_name)
        history = await self.wait_for_prompt(job_id, prompt_id, client_id)
        output = _video_output(history)
        if not output:
            raise RuntimeError("ComfyUI completed without returning a video output")
        output = await self.prepare_output(job, output)
        await self.complete_job(job_id, output)

    async def prepare_output(self, job: dict[str, Any], output: dict[str, str]) -> dict[str, str]:
        if str(job.get("output_format") or "video") != "gif":
            return output

        source_path = _safe_local_path(
            COMFY_OUTPUT_ROOT, output.get("subfolder"), output["filename"]
        )
        gif_path = source_path.with_suffix(".gif")
        if not gif_path.is_file() or gif_path.stat().st_size == 0:
            if not source_path.is_file():
                raise RuntimeError("The generated video is unavailable for GIF conversion")
            await self.update_progress(str(job["id"]), 97)
            gif_executable, gif_message = _gif_conversion_status()
            if not gif_executable:
                raise RuntimeError(gif_message)

            temporary_gif_path = gif_path.with_name(f"{gif_path.stem}.part.gif")
            temporary_gif_path.unlink(missing_ok=True)
            process = await asyncio.to_thread(
                subprocess.run,
                [
                    gif_executable,
                    "-y",
                    "-i",
                    str(source_path),
                    "-filter_complex",
                    "fps=12,split[frames][palette_input];[palette_input]palettegen=max_colors=192[palette];[frames][palette]paletteuse=dither=sierra2_4a",
                    "-loop",
                    "0",
                    str(temporary_gif_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if process.returncode != 0 or not temporary_gif_path.is_file() or temporary_gif_path.stat().st_size == 0:
                temporary_gif_path.unlink(missing_ok=True)
                detail = process.stderr.decode("utf-8", errors="replace")[-800:]
                raise RuntimeError(f"GIF conversion failed: {detail}")
            temporary_gif_path.replace(gif_path)
        source_path.unlink(missing_ok=True)

        return {
            "filename": gif_path.name,
            "subfolder": output.get("subfolder") or "",
            "type": output.get("type") or "output",
        }

    async def submit_prompt(
        self, job_id: str, workflow: dict[str, Any], first_name: str | None, last_name: str | None
    ) -> tuple[str, str]:
        assert self.http is not None
        client_id = str(uuid.uuid4())
        async with self.http.post(
            f"{COMFY_BASE_URL}/prompt", json={"prompt": workflow, "client_id": client_id}
        ) as response:
            payload = await response.json(content_type=None)
            if response.status != 200 or not payload.get("prompt_id"):
                raise RuntimeError(f"ComfyUI rejected workflow ({response.status}): {payload}")
        prompt_id = str(payload["prompt_id"])
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE video_generation_queue
                    SET comfy_prompt_id = %s, comfy_client_id = %s,
                        first_input_name = %s, last_input_name = %s,
                        first_frame = NULL, last_frame = NULL, progress = 3, updated_at = NOW()
                    WHERE id = %s AND status = 'processing'
                    """,
                    (prompt_id, client_id, first_name, last_name, job_id),
                )
                await cur.execute("UPDATE video_generation_references SET media = NULL WHERE job_id = %s", (job_id,))
            await conn.commit()
        return prompt_id, client_id

    async def _history(self, prompt_id: str) -> dict[str, Any] | None:
        assert self.http is not None
        async with self.http.get(f"{COMFY_BASE_URL}/history/{prompt_id}") as response:
            if response.status != 200:
                return None
            payload = await response.json(content_type=None)
        entry = payload.get(prompt_id) if isinstance(payload, dict) else None
        return entry if isinstance(entry, dict) else None

    async def wait_for_prompt(
        self, job_id: str, prompt_id: str, client_id: str | None = None
    ) -> dict[str, Any]:
        assert self.http is not None
        parsed = urlparse(COMFY_BASE_URL)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = f"{scheme}://{parsed.netloc}/ws?clientId={client_id or uuid.uuid4()}"
        deadline = asyncio.get_running_loop().time() + GENERATION_TIMEOUT_SECONDS
        progress = 3
        try:
            async with self.http.ws_connect(ws_url, heartbeat=30, receive_timeout=30) as websocket:
                while asyncio.get_running_loop().time() < deadline:
                    history = await self._history(prompt_id)
                    if history:
                        output = _video_output(history)
                        status = history.get("status") or {}
                        if output:
                            return history
                        if status.get("status_str") == "error" or status.get("completed") is False:
                            messages = status.get("messages") or []
                            raise RuntimeError(f"ComfyUI execution failed: {messages[-1] if messages else status}")
                    try:
                        message = await websocket.receive(timeout=15)
                    except asyncio.TimeoutError:
                        continue
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    data = event.get("data") or {}
                    if event.get("type") == "progress" and str(data.get("prompt_id")) == prompt_id:
                        value = int(data.get("value") or 0)
                        maximum = max(1, int(data.get("max") or 1))
                        progress = max(progress, min(96, 5 + int(value / maximum * 90)))
                        await self.update_progress(job_id, progress)
                    if event.get("type") == "execution_error" and str(data.get("prompt_id")) == prompt_id:
                        raise RuntimeError(str(data.get("exception_message") or "ComfyUI execution failed"))
        except aiohttp.WSServerHandshakeError:
            logger.warning("ComfyUI websocket unavailable; falling back to history polling")

        while asyncio.get_running_loop().time() < deadline:
            history = await self._history(prompt_id)
            if history and _video_output(history):
                return history
            await asyncio.sleep(5)
        raise RuntimeError("Video generation timed out")

    async def update_progress(self, job_id: str, progress: int) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE video_generation_queue
                    SET progress = GREATEST(progress, %s), worker_heartbeat_at = NOW(), updated_at = NOW()
                    WHERE id = %s AND status = 'processing'
                    """,
                    (progress, job_id),
                )
            await conn.commit()

    async def complete_job(self, job_id: str, output: dict[str, str]) -> None:
        mime = mimetypes.guess_type(output["filename"])[0] or "video/mp4"
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE video_generation_queue
                    SET status = 'completed', progress = 100, completed_at = NOW(),
                        expires_at = NOW() + INTERVAL '24 hours', output_filename = %s,
                        output_subfolder = %s, output_type = %s, output_mime = %s,
                        first_frame = NULL, last_frame = NULL, error_message = NULL, updated_at = NOW()
                    WHERE id = %s AND status = 'processing'
                    """,
                    (output["filename"], output["subfolder"], output["type"], mime, job_id),
                )
                await cur.execute("UPDATE video_generation_references SET media = NULL WHERE job_id = %s", (job_id,))
            await conn.commit()
        logger.info("Video job %s completed: %s", job_id, output["filename"])

    async def fail_job(self, job_id: str | None, error: str) -> None:
        if not job_id:
            return
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE video_generation_queue
                    SET status = 'failed', completed_at = NOW(), expires_at = NOW() + INTERVAL '24 hours',
                        first_frame = NULL, last_frame = NULL, error_message = %s, updated_at = NOW()
                    WHERE id = %s AND status IN ('pending', 'processing')
                    """,
                    (error[:1000], job_id),
                )
                if cur.rowcount:
                    await cur.execute(
                        "SELECT * FROM safe_refund_video_credits(%s, %s)",
                        (job_id, f"Video generation failed: {error[:400]}"),
                    )
                    await cur.execute("UPDATE video_generation_references SET media = NULL WHERE job_id = %s", (job_id,))
            await conn.commit()

    async def _comfy_queue_prompt_ids(self) -> set[str]:
        assert self.http is not None
        async with self.http.get(f"{COMFY_BASE_URL}/queue") as response:
            if response.status != 200:
                return set()
            payload = await response.json(content_type=None)
        ids: set[str] = set()
        for key in ("queue_running", "queue_pending"):
            for entry in payload.get(key) or []:
                if isinstance(entry, list) and len(entry) > 1:
                    ids.add(str(entry[1]))
        return ids

    async def reconcile_processing_jobs(self, *, skip_health_check: bool = False) -> None:
        if not skip_health_check:
            healthy, message = await self.comfy_health()
            if not healthy:
                logger.warning("Processing-job reconciliation deferred: %s", message)
                return
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM video_generation_queue WHERE status = 'processing' ORDER BY started_at"
                )
                jobs = [dict(row) for row in await cur.fetchall()]
        queued_ids = await self._comfy_queue_prompt_ids()
        for job in jobs:
            job_id = str(job["id"])
            prompt_id = str(job.get("comfy_prompt_id") or "")
            if prompt_id:
                history = await self._history(prompt_id)
                output = _video_output(history) if history else None
                if output:
                    output = await self.prepare_output(job, output)
                    await self.complete_job(job_id, output)
                    continue
                if prompt_id in queued_ids:
                    self.current_job_id = job_id
                    try:
                        history = await self.wait_for_prompt(job_id, prompt_id, job.get("comfy_client_id"))
                        output = _video_output(history)
                        if output:
                            output = await self.prepare_output(job, output)
                            await self.complete_job(job_id, output)
                            continue
                    except Exception as exc:
                        await self.fail_job(job_id, f"Could not recover interrupted job: {exc}")
                    finally:
                        self.current_job_id = None
                    continue
                await self.fail_job(job_id, "Interrupted job could not be found in ComfyUI queue or history")
            elif job.get("first_frame") or (job.get("generation_mode") == "ref2v" and any(ref.get("media") for ref in await self.reference_inputs(job_id))):
                async with self.pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            UPDATE video_generation_queue
                            SET status = 'pending', started_at = NULL, progress = 0, updated_at = NOW()
                            WHERE id = %s AND status = 'processing'
                            """,
                            (job_id,),
                        )
                    await conn.commit()
            else:
                await self.fail_job(job_id, "Interrupted job has no recoverable frame data")

    async def cleanup_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=600)
            except asyncio.TimeoutError:
                await self.cleanup_expired_files()

    async def cleanup_expired_files(self) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT * FROM video_generation_queue
                    WHERE (status = 'completed' AND expires_at <= NOW())
                       OR (status IN ('failed', 'cancelled', 'expired') AND updated_at <= NOW() - INTERVAL '24 hours')
                    FOR UPDATE SKIP LOCKED
                    """
                )
                jobs = [dict(row) for row in await cur.fetchall()]
                for job in jobs:
                    try:
                        if job.get("output_filename"):
                            output = _safe_local_path(
                                COMFY_OUTPUT_ROOT, job.get("output_subfolder"), str(job["output_filename"])
                            )
                            output.unlink(missing_ok=True)
                        await cur.execute("SELECT input_name FROM video_generation_references WHERE job_id = %s", (job["id"],))
                        reference_names = [row["input_name"] for row in await cur.fetchall()]
                        for input_name in (job.get("first_input_name"), job.get("last_input_name"), *reference_names):
                            if input_name:
                                input_path = _safe_local_path(COMFY_INPUT_ROOT, None, str(input_name))
                                input_path.unlink(missing_ok=True)
                        await cur.execute("DELETE FROM video_generation_references WHERE job_id = %s", (job["id"],))
                        await cur.execute(
                            """
                            UPDATE video_generation_queue
                            SET status = 'expired', first_frame = NULL, last_frame = NULL,
                                first_frame_thumbnail = NULL, last_frame_thumbnail = NULL,
                                output_filename = NULL, output_subfolder = NULL, updated_at = NOW()
                            WHERE id = %s
                            """,
                            (job["id"],),
                        )
                    except (OSError, ValueError) as exc:
                        logger.error("Cleanup failed for video job %s: %s", job["id"], exc)
            await conn.commit()


async def main() -> None:
    worker = VideoWorker()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, worker.request_stop)
    await worker.start()


async def health_check() -> None:
    worker = VideoWorker()
    worker.http = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60, connect=10), trust_env=True
    )
    try:
        storage_ok, storage_message = worker._storage_status()
        comfy_ok, comfy_message = await worker.comfy_health(force=True)
        logger.info("Storage: %s", storage_message)
        logger.info("ComfyUI: %s", comfy_message)
        if not storage_ok or not comfy_ok:
            raise SystemExit(1)
    finally:
        await worker.http.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(health_check() if "--health-check" in sys.argv else main())
