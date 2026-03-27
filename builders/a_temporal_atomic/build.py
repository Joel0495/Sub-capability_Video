"""Builder for sub-capability A: Temporal Atomic.

Uses VideoR1Adapter for RL-native temporal atomic reasoning data.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.video_r1 import VideoR1Adapter
from builders.base import BaseBuilder


class TemporalAtomicBuilder(BaseBuilder):
    capability = SubCapability.TEMPORAL_ATOMIC
    capability_name = "a_temporal_atomic"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            VideoR1Adapter(data_root=self.data_root),
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
