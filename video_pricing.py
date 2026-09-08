"""REF2V quotes calibrated to the September 2026 capped-reference benchmarks.

All durations keep the existing FL2V price anchor. Intermediate counts/durations
are interpolated; outputs longer than 10 seconds extrapolate the measured trend.
See VIDEO_GENERATION.md for the calibration and pricing examples.
"""
from __future__ import annotations

import math
from video_core import (VIDEO_MAX_REFERENCE_IMAGES, VIDEO_MAX_REFERENCE_VIDEOS,
                        VIDEO_REFERENCE_MIN_SECONDS, VIDEO_REFERENCE_MAX_SECONDS,
                        VideoInputError, price_for_duration)

VIDEO_PRICING_VERSION = "ref2v-04mp-v1"
_FL_SECONDS_5 = 25.298
_FL_SECONDS_10 = 58.514
# Marginal execution seconds above FL2V, indexed by image count / usable frames.
_IMAGE_SECONDS_5 = ((0, 0), (1, 0), (3, 10.739), (6, 10.998), (9, 16.796))
_VIDEO_SECONDS_5 = ((0, 0), (22, 5.602), (124, 44.387), (248, 97.349), (372, 165.327))
# Full clips are decoded before the model clips their usable video frames.
_EXCESS_SECOND_COST = (75.687 - 69.685) / (15 - 124 / 24)
_VIDEO_SECONDS_10 = ((0, 0), (124, 107.082 - _FL_SECONDS_10),
                     (243, 174.497 - _FL_SECONDS_10 - (15 - 243 / 24) * _EXCESS_SECOND_COST))
_IMAGE_TIME_SCALE_10 = (75.929 - _FL_SECONDS_10) / 16.796
_MIXED_IMAGE_SECONDS_5 = (75.559 - 69.685) / 3


def _interpolate(points: tuple, value: float) -> float:
    for left, right in zip(points, points[1:]):
        if value <= right[0]:
            break
    return left[1] + (value - left[0]) * (right[1] - left[1]) / (right[0] - left[0])


def video_price_quote(duration_seconds: int, generation_mode: str = "fl2v",
                      reference_image_count: int = 0,
                      reference_video_seconds: list[float] | None = None) -> dict:
    base = price_for_duration(duration_seconds)
    if generation_mode not in {"fl2v", "ref2v"}:
        raise VideoInputError("Generation mode must be fl2v or ref2v.")
    if (isinstance(reference_image_count, bool) or not isinstance(reference_image_count, int)
            or not 0 <= reference_image_count <= VIDEO_MAX_REFERENCE_IMAGES):
        raise VideoInputError("Use up to 9 reference images.")
    durations = [] if reference_video_seconds is None else reference_video_seconds
    if not isinstance(durations, list) or len(durations) > VIDEO_MAX_REFERENCE_VIDEOS:
        raise VideoInputError("Use up to 3 reference videos.")
    for seconds in durations:
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or not VIDEO_REFERENCE_MIN_SECONDS <= seconds <= VIDEO_REFERENCE_MAX_SECONDS):
            raise VideoInputError("Reference videos must be between 0.25 and 15 seconds long.")
    if generation_mode == "fl2v" and (reference_image_count or durations):
        raise VideoInputError("Reference media belongs in Use references mode.")

    # Match the workflow's upward 17k+5 output grid and the reference node's
    # downward grid after clipping. Round duration metadata to FFmpeg precision.
    output_frames = int(duration_seconds) * 24
    output_frames += (5 - output_frames % 17) % 17
    frames = [max(5, (min(round(round(seconds, 2) * 24), output_frames) - 5) // 17 * 17 + 5)
              for seconds in durations]
    effective_seconds = [count / 24 for count in frames]
    extra = 0.0
    if generation_mode == "ref2v":
        progress = (int(duration_seconds) - 5) / 5
        fl_seconds = _FL_SECONDS_5 + progress * (_FL_SECONDS_10 - _FL_SECONDS_5)
        image_scale = 1 + progress * (_IMAGE_TIME_SCALE_10 - 1)
        if durations:
            workload = sum(frames)
            five = _interpolate(_VIDEO_SECONDS_5, workload)
            ten = _interpolate(_VIDEO_SECONDS_10, workload)
            extra = five + progress * (ten - five)
            extra += sum(max(0, round(seconds, 2) - output_frames / 24) for seconds in durations) * _EXCESS_SECOND_COST
            extra += reference_image_count * _MIXED_IMAGE_SECONDS_5 * image_scale
        else:
            extra = _interpolate(_IMAGE_SECONDS_5, reference_image_count) * image_scale
        extra = base * extra / fl_seconds
    total = max(base, math.ceil((base + extra - 1e-8) / 10) * 10)
    return dict(pricing_version=VIDEO_PRICING_VERSION, credit_cost=total,
                base_cost=base, reference_cost=total-base,
                effective_video_seconds=[round(seconds, 3) for seconds in effective_seconds])
