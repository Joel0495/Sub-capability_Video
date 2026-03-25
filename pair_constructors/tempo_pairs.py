"""TEMPO-style contrastive pair constructor.

Creates temporally-perturbed variants of canonical samples for
anti-shortcut training. Perturbation types:
- reverse_order: Reverse temporal order of events
- shuffle_segments: Randomly shuffle video segments
- speed_change: Alter playback speed

These pairs feed into capability K (anti_shortcut_contrast) and
can augment any capability's RL-ready data.
"""

from typing import Iterator
from schema.canonical import (
    CanonicalSample,
    SubCapability,
    BuildType,
    Perturbation,
    DataInfo,
    BuildInfo,
)


PERTURBATION_TYPES = [
    "reverse_order",
    "shuffle_segments",
    "speed_change",
    "frame_drop",
    "sparse_frame_ablation",
]


class TEMPOPairConstructor:
    """Construct temporally-perturbed contrastive pairs from canonical samples.

    TEMPO methodology:
    1. Take a clean canonical sample with correct answer
    2. Apply a temporal perturbation to the video reference
    3. The perturbed version may change the correct answer
    4. Emit both as a paired sample for anti-shortcut training
    """

    def __init__(self, perturbation_type: str = "reverse_order"):
        if perturbation_type not in PERTURBATION_TYPES:
            raise ValueError(
                f"Unknown perturbation: {perturbation_type}. "
                f"Choose from: {PERTURBATION_TYPES}"
            )
        self.perturbation_type = perturbation_type

    def build_pairs(
        self,
        samples: list[CanonicalSample],
    ) -> Iterator[tuple[CanonicalSample, CanonicalSample]]:
        """Build contrastive pairs from clean canonical samples.

        Yields:
            (clean_sample, perturbed_sample) tuples

        TODO: Implement actual video perturbation logic.
        Currently yields stub pairs with updated metadata.
        """
        for sample in samples:
            perturbed = self._create_perturbed(sample)
            if perturbed is not None:
                yield (sample, perturbed)

    def _create_perturbed(
        self, sample: CanonicalSample
    ) -> CanonicalSample | None:
        """Create a perturbed version of a sample.

        TODO: Implement actual perturbation logic:
        - reverse_order: Reverse the temporal ordering in the question/answer
        - shuffle_segments: Create a shuffled video variant
        - speed_change: Modify decode_fps to simulate speed change
        - frame_drop: Drop frames to create sparse variant
        - sparse_frame_ablation: Keep only every Nth frame
        """
        original_id = sample.data_info.data_id
        perturbed_id = f"{original_id}_perturbed_{self.perturbation_type}"

        # Determine if answer changes under perturbation
        answer_changed = self.perturbation_type in [
            "reverse_order",
            "shuffle_segments",
        ]

        # Create perturbed data_info
        perturbed_data_info = sample.data_info.model_copy(deep=True)
        perturbed_data_info.data_id = perturbed_id
        perturbed_data_info.build_info = BuildInfo(
            build_type=BuildType.PAIRED,
            video_path=sample.data_info.build_info.video_path,
        )
        perturbed_data_info.perturbation = Perturbation(
            paired_data_id=original_id,
            type=self.perturbation_type,
            answer_changed=answer_changed,
            difficulty="medium",
        )

        # Also update original with perturbation link
        if sample.data_info.perturbation is None:
            sample.data_info.perturbation = Perturbation(
                paired_data_id=perturbed_id,
                type=self.perturbation_type,
                answer_changed=answer_changed,
            )

        # For now, return sample with updated metadata
        # Actual video perturbation would modify the video_url or
        # add perturbation parameters
        return CanonicalSample(
            messages=sample.messages,  # TODO: modify answer if answer_changed
            graders=sample.graders,
            data_info=perturbed_data_info,
            extra_info=sample.extra_info,
        )
