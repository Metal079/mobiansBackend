from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


VIDEO_PRICES: dict[int, int] = {
    5: 80,
    6: 120,
    7: 150,
    8: 180,
    9: 210,
    10: 240,
    11: 270,
    12: 300,
    13: 330,
    14: 360,
    15: 400,
}
VIDEO_ASPECTS: dict[str, dict[str, Any]] = {
    "square": {"width": 640, "height": 640, "comfy_value": "1:1 (Square)"},
    "landscape": {"width": 768, "height": 512, "comfy_value": "3:2 (Photo)"},
    "portrait": {"width": 512, "height": 768, "comfy_value": "2:3 (Portrait Photo)"},
}
VIDEO_MAX_FRAME_BYTES = 15 * 1024 * 1024
VIDEO_MAX_IMAGE_PIXELS = 40_000_000
# The UI allows 8,000 user-entered characters and may append a camera command.
VIDEO_MAX_PROMPT_CHARS = 8_020
VIDEO_MAX_AUDIO_PROMPT_CHARS = 4_000
VIDEO_MAX_REFERENCE_IMAGES = 9
VIDEO_MAX_REFERENCE_VIDEOS = 3
VIDEO_MAX_REFERENCE_VIDEO_BYTES = 50 * 1024 * 1024
VIDEO_MAX_REFERENCE_TOTAL_BYTES = 150 * 1024 * 1024
VIDEO_REFERENCE_MIN_SECONDS = 0.25
VIDEO_REFERENCE_MAX_SECONDS = 15

WORKFLOW_PATH = Path(__file__).resolve().parent / "video_workflows" / "minimax_h3_hybrid_api.json"
WATERMARK_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Roboto-Medium.ttf"
WATERMARK_TEXT = "Mobians.ai"
WATERMARK_OPACITY = 128


class VideoInputError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizedVideoFrame:
    image_bytes: bytes
    mime_type: str
    thumbnail_bytes: bytes
    thumbnail_mime_type: str
    width: int
    height: int


def price_for_duration(duration_seconds: int) -> int:
    if isinstance(duration_seconds, bool):
        raise VideoInputError("Duration must be a whole number from 5 through 15 seconds.")
    try:
        normalized = int(duration_seconds)
        if isinstance(duration_seconds, float) and not duration_seconds.is_integer():
            raise ValueError
        if isinstance(duration_seconds, str) and duration_seconds.strip() != str(normalized):
            raise ValueError
        return VIDEO_PRICES[normalized]
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoInputError("Duration must be a whole number from 5 through 15 seconds.") from exc


def aspect_settings(aspect_ratio: str) -> dict[str, Any]:
    try:
        return VIDEO_ASPECTS[str(aspect_ratio)]
    except KeyError as exc:
        raise VideoInputError("Aspect ratio must be square, landscape, or portrait.") from exc


def normalize_prompt(prompt: str, audio_prompt: str | None = None) -> tuple[str, str | None]:
    normalized_prompt = str(prompt or "").strip()
    normalized_audio = str(audio_prompt or "").strip() or None
    if not normalized_prompt:
        raise VideoInputError("A video prompt is required.")
    if len(normalized_prompt) > VIDEO_MAX_PROMPT_CHARS:
        raise VideoInputError(f"Video prompt must be {VIDEO_MAX_PROMPT_CHARS} characters or fewer.")
    if normalized_audio and len(normalized_audio) > VIDEO_MAX_AUDIO_PROMPT_CHARS:
        raise VideoInputError(f"Audio direction must be {VIDEO_MAX_AUDIO_PROMPT_CHARS} characters or fewer.")
    return normalized_prompt, normalized_audio


def normalize_video_frame(data: bytes) -> NormalizedVideoFrame:
    if not data:
        raise VideoInputError("The selected frame is empty.")
    if len(data) > VIDEO_MAX_FRAME_BYTES:
        raise VideoInputError("Each frame must be 15 MB or smaller.")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = VIDEO_MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise VideoInputError("Frames must be JPEG, PNG, or WebP images.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width < 64 or image.height < 64:
                raise VideoInputError("Frames must be at least 64×64 pixels.")
            return _encode_video_frame(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise VideoInputError("The selected frame is not a valid supported image.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def _encode_video_frame(image: Image.Image) -> NormalizedVideoFrame:
    normalized = image.convert("RGB")
    output = io.BytesIO()
    normalized.save(output, format="PNG", optimize=True)

    thumbnail = normalized.copy()
    thumbnail.thumbnail((384, 384), Image.Resampling.LANCZOS)
    thumb_output = io.BytesIO()
    thumbnail.save(thumb_output, format="WEBP", quality=82, method=4)
    return NormalizedVideoFrame(
        image_bytes=output.getvalue(),
        mime_type="image/png",
        thumbnail_bytes=thumb_output.getvalue(),
        thumbnail_mime_type="image/webp",
        width=normalized.width,
        height=normalized.height,
    )


def video_frame_needs_watermark(source: str | None) -> bool:
    normalized = str(source or "upload").strip().lower()
    if normalized not in {"upload", "history"}:
        raise VideoInputError("Frame source must be upload or history.")
    return normalized == "upload"


def watermark_video_frame(frame: NormalizedVideoFrame) -> NormalizedVideoFrame:
    with Image.open(io.BytesIO(frame.image_bytes)) as source:
        image = source.convert("RGBA")

    font_size = max(int(image.width * 0.05), 10)
    font = ImageFont.truetype(str(WATERMARK_FONT_PATH), font_size)
    margin_x = int(image.width * 0.02)
    margin_y = int(image.height * 0.01)
    stroke_width = max(2, int(font_size / 20))

    watermark = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark)
    draw.text(
        (margin_x, margin_y),
        WATERMARK_TEXT,
        font=font,
        fill=(255, 255, 255, WATERMARK_OPACITY),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, WATERMARK_OPACITY),
    )
    return _encode_video_frame(Image.alpha_composite(image, watermark))


def load_workflow_template(path: Path = WORKFLOW_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as workflow_file:
        return json.load(workflow_file)


def validate_generation_inputs(mode: str, first: bool, last: bool, images: int, videos: int) -> None:
    if mode not in {"fl2v", "ref2v"}:
        raise VideoInputError("Generation mode must be fl2v or ref2v.")
    if images > VIDEO_MAX_REFERENCE_IMAGES or videos > VIDEO_MAX_REFERENCE_VIDEOS:
        raise VideoInputError("Use up to 9 reference images and 3 reference videos.")
    if mode == "fl2v":
        if not first:
            raise VideoInputError("A first frame is required in Animate an image mode.")
        if images or videos:
            raise VideoInputError("Reference media belongs in Use references mode.")
    else:
        if first or last:
            raise VideoInputError("First and last frames belong in Animate an image mode.")
        if not images and not videos:
            raise VideoInputError("Add at least one reference image or video.")


def build_minimax_h3_workflow(
    *,
    first_input_name: str | None = None,
    last_input_name: str | None = None,
    prompt: str,
    audio_prompt: str | None,
    duration_seconds: int,
    aspect_ratio: str,
    seed: int,
    job_id: str,
    disable_sound: bool = False,
    generation_mode: str = "fl2v",
    reference_images: list[str] | None = None,
    reference_videos: list[dict[str, Any]] | None = None,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    price_for_duration(duration_seconds)
    aspect = aspect_settings(aspect_ratio)
    prompt, audio_prompt = normalize_prompt(prompt, audio_prompt)
    images, videos = reference_images or [], reference_videos or []
    validate_generation_inputs(generation_mode, bool(first_input_name), bool(last_input_name), len(images), len(videos))
    workflow = copy.deepcopy(template if template is not None else load_workflow_template())
    conditioning_id = "164" if generation_mode == "fl2v" else "136"
    conditioning_inputs = workflow[conditioning_id]["inputs"]
    if generation_mode == "fl2v":
        workflow["162"]["inputs"]["image"] = first_input_name
        conditioning_inputs["first_frame"] = ["162", 0]
        if last_input_name:
            workflow["163"]["inputs"]["image"] = last_input_name
            conditioning_inputs["last_frame"] = ["163", 0]
        else:
            conditioning_inputs.pop("last_frame", None)
    else:
        # No example media may reach validation or generation.
        for key in list(conditioning_inputs):
            if key.startswith(("ref_images.", "ref_videos.", "ref_video_audios.", "ref_audios.")):
                del conditioning_inputs[key]
        for index, filename in enumerate(images):
            node_id = str(200 + index)
            workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            conditioning_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        for index, reference in enumerate(videos):
            loader, components = str(220 + index * 2), str(221 + index * 2)
            workflow[loader] = {"class_type": "LoadVideo", "inputs": {"file": reference["name"]}}
            workflow[components] = {"class_type": "GetVideoComponents", "inputs": {"video": [loader, 0]}}
            conditioning_inputs[f"ref_videos.ref_video_{index}"] = [components, 0]
            if reference.get("use_audio"):
                conditioning_inputs[f"ref_video_audios.ref_video_audio_{index}"] = [components, 1]

    combined_prompt = prompt
    if audio_prompt and not disable_sound:
        combined_prompt = f"{prompt}\n\nAudio: {audio_prompt}"
    conditioning_inputs["prompt"] = combined_prompt
    workflow["126"]["inputs"]["conditioning"] = [conditioning_id, 0]
    workflow["125"]["inputs"]["latent_image"] = [conditioning_id, 1]
    workflow["132"]["inputs"]["value"] = int(duration_seconds)
    workflow["129"]["inputs"]["noise_seed"] = int(seed)
    # ResolutionSelector rounds some aspect ratios up to 800x544 at 0.4 MP.
    # Preserve the website's exact existing presets and its displayed metadata.
    conditioning_inputs["width"] = aspect["width"]
    conditioning_inputs["height"] = aspect["height"]
    if disable_sound:
        workflow["130"]["inputs"].pop("audio", None)
    workflow["92"]["inputs"]["filename_prefix"] = f"video/MiniMax_H3/{job_id}"
    # Submit only output dependencies: no inactive switches or media loaders.
    reachable: set[str] = set()
    def visit(node_id: str) -> None:
        if node_id in reachable:
            return
        reachable.add(node_id)
        for value in workflow[node_id]["inputs"].values():
            if isinstance(value, list):
                visit(value[0])
    visit("92")
    return {key: value for key, value in workflow.items() if key in reachable}


def effective_service_state(
    *,
    feature_enabled: bool,
    desired_state: str,
    worker_status: str | None,
    heartbeat_age_seconds: float | None,
    heartbeat_timeout_seconds: float = 45.0,
) -> str:
    if not feature_enabled:
        return "disabled"
    if desired_state in {"draining", "maintenance"}:
        return desired_state
    if heartbeat_age_seconds is None or heartbeat_age_seconds > heartbeat_timeout_seconds:
        return "offline"
    if worker_status == "low_storage":
        return "low_storage"
    if worker_status not in {"idle", "processing", "healthy"}:
        return "offline"
    return "available"
