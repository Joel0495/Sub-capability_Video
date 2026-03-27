"""Builder for sub-capability B: Temporal Count & Order.

Uses VideoR1Adapter for RL-native temporal counting and ordering data.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.video_r1 import VideoR1Adapter
from builders.base import BaseBuilder


class TemporalCountOrderBuilder(BaseBuilder):
    capability = SubCapability.TEMPORAL_COUNT_ORDER
    capability_name = "b_temporal_count_order"

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
