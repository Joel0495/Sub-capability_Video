"""Builder for sub-capability I: Multimodal Audio-Visual Fusion.

Uses Ego4DAdapter for audio-visual fusion data from egocentric video.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.ego4d import Ego4DAdapter
from builders.base import BaseBuilder


class MultimodalAVFusionBuilder(BaseBuilder):
    capability = SubCapability.MULTIMODAL_AV_FUSION
    capability_name = "i_multimodal_av_fusion"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            Ego4DAdapter(data_root=self.data_root),
        ]

    def build_samples(self, adapter: BaseAdapter, split: str) -> list[CanonicalSample]:
        samples = []
        for raw in adapter.iterate_raw(split):
            sample = adapter.to_canonical(
                raw,
                sub_capability=self.capability,
                task_type=TaskType.MCQ,
            )
            samples.append(sample)
        return samples
