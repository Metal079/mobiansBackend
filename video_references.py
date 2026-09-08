"""Reference-video validation and normalization, shared by API tests and uploads."""
from __future__ import annotations

import io
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from video_core import (
    VIDEO_MAX_REFERENCE_VIDEO_BYTES, VIDEO_REFERENCE_MIN_SECONDS,
    VIDEO_REFERENCE_MAX_SECONDS, NormalizedVideoFrame, VideoInputError,
    normalize_video_frame,
)


# Match the largest existing 0.4 MP output preset. MiniMax rounds reference
# dimensions to multiples of 32, so the cap must also cover that canvas.
VIDEO_REFERENCE_MAX_PIXELS = 640 * 640
VIDEO_REFERENCE_MAX_SIDE = 1280


def _reference_video_bounds(width: int, height: int) -> tuple[int, int]:
    canvas_width = max(32, round(width / 32) * 32)
    canvas_height = max(32, round(height / 32) * 32)
    if (max(width, height) <= VIDEO_REFERENCE_MAX_SIDE
            and width * height <= VIDEO_REFERENCE_MAX_PIXELS
            and canvas_width * canvas_height <= VIDEO_REFERENCE_MAX_PIXELS):
        return width, height
    scale = min(1.0, math.sqrt(VIDEO_REFERENCE_MAX_PIXELS / (width * height)),
                VIDEO_REFERENCE_MAX_SIDE / width, VIDEO_REFERENCE_MAX_SIDE / height)
    # Fit inside an aligned box, keeping the source aspect ratio in FFmpeg.
    # Rounding the resulting frames in ComfyUI cannot exceed this box.
    return (max(32, math.floor(width * scale / 32) * 32),
            max(32, math.floor(height * scale / 32) * 32))


@dataclass(frozen=True)
class NormalizedReferenceVideo:
    media: bytes
    thumbnail: NormalizedVideoFrame
    duration: float
    has_audio: bool


def normalize_reference_video(data: bytes, use_audio: bool) -> NormalizedReferenceVideo:
    if not data or len(data) > VIDEO_MAX_REFERENCE_VIDEO_BYTES:
        raise VideoInputError("Each reference video must be nonempty and 50 MB or smaller.")
    # Accept file containers, never playlists or arbitrary FFmpeg input URLs.
    if not (data[4:8] == b"ftyp" or data[:4] == b"\x1a\x45\xdf\xa3"):
        raise VideoInputError("Reference videos must be MP4, MOV, or WebM files.")
    try:
        with tempfile.TemporaryDirectory(prefix="mobians-video-ref-") as directory:
            source, output = Path(directory) / "source.video", Path(directory) / "reference.mp4"
            source.write_bytes(data)
            reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24", input_params=["-protocol_whitelist", "file,pipe"], output_params=["-frames:v", "1"])
            try:
                metadata = next(reader)
                width, height = metadata["size"]
                duration = float(metadata.get("duration") or 0)
                if not math.isfinite(duration) or not VIDEO_REFERENCE_MIN_SECONDS <= duration <= VIDEO_REFERENCE_MAX_SECONDS:
                    raise VideoInputError("Reference videos must be between 0.25 and 15 seconds long.")
                if width < 64 or height < 64 or width * height > 8_294_400:
                    raise VideoInputError("Reference videos must be at least 64 pixels per side and no larger than 4K.")
                frame = next(reader)
            finally:
                reader.close()
            image = Image.frombytes("RGB", (width, height), frame)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            thumbnail = normalize_video_frame(buffer.getvalue())
            # Normalize before storage/upload to ComfyUI so a high-resolution
            # input cannot increase GPU reference encoding or sampling costs.
            bound_width, bound_height = _reference_video_bounds(width, height)
            filters = (f"fps=24,scale=w={bound_width}:h={bound_height}:"
                       "force_original_aspect_ratio=decrease:force_divisible_by=2")
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error", "-xerror",
                       "-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:v:0",
                       "-vf", filters, "-t", "15", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]
            # imageio's metadata includes an audio codec only when an audio stream exists.
            has_audio = bool(metadata.get("audio_codec"))
            if use_audio and has_audio:
                command += ["-map", "0:a:0?", "-c:a", "aac", "-ar", "32000", "-ac", "2"]
            else:
                command += ["-an"]
            command += ["-movflags", "+faststart", str(output)]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90, check=False)
            if result.returncode or not output.is_file() or not output.stat().st_size:
                raise VideoInputError("The reference video could not be decoded. Try another MP4, MOV, or WebM file.")
            if output.stat().st_size > VIDEO_MAX_REFERENCE_VIDEO_BYTES:
                raise VideoInputError("The normalized reference video is too large. Use a shorter clip.")
            return NormalizedReferenceVideo(output.read_bytes(), thumbnail, duration, use_audio and has_audio)
    except VideoInputError:
        raise
    except (OSError, RuntimeError, ValueError, StopIteration, subprocess.TimeoutExpired) as exc:
        raise VideoInputError("The reference video could not be read. Try another MP4, MOV, or WebM file.") from exc
