"""TPO pair constructor stub.

Creates preference pairs by contrasting responses to intact video
vs. sparse-downsampled video. The "chosen" response comes from the
model seeing the full video; the "rejected" response from the model
seeing only sparsely sampled frames.
"""

from __future__ import annotations

from typing import Iterator

from schema.canonical import CanonicalSample


def build_tpo_pairs(
    samples: list[CanonicalSample],
    sparse_fps: float = 0.5,
    min_duration_s: float = 5.0,
) -> Iterator[dict]:
    """Build TPO preference pairs from canonical samples.

    For each sample with a video longer than min_duration_s:
    1. Create a "chosen" variant using the intact video (original fps)
    2. Create a "rejected" variant using sparse-downsampled video (sparse_fps)
    3. Yield a preference pair dict

    Args:
        samples: List of canonical samples to build pairs from.
        sparse_fps: FPS for the sparse-downsampled variant.
        min_duration_s: Minimum video duration to be eligible for pairing.

    Yields:
        Preference pair dicts with keys: chosen, rejected, data_id, pair_type
    """
    # TODO: Implement pair construction
    # Stub: yields nothing for now
    for sample in samples:
        video_profile = (
            sample.data_info.video_profile if sample.data_info.video_profile else None
        )
        if video_profile is None:
            continue
        duration = video_profile.duration_s or 0.0
        if duration < min_duration_s:
            continue

        # Placeholder: actual implementation would:
        # 1. Run inference with intact video -> chosen response
        # 2. Run inference with sparse video (sparse_fps) -> rejected response
        # 3. Yield the pair
        _ = sparse_fps  # suppress unused warning
        yield {
            "chosen": sample.model_dump(),
            "rejected": sample.model_dump(),  # placeholder
            "data_id": f"tpo_{sample.data_info.data_id}",
            "pair_type": "tpo_intact_vs_sparse",
        }
