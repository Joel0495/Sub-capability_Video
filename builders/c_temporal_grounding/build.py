"""Builder for sub-capability C: Temporal Grounding.

Uses QVHighlightsAdapter and ActivityNetCaptionsAdapter for temporal
grounding data.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.qvhighlights import QVHighlightsAdapter
from source_adapters.activitynet_captions import ActivityNetCaptionsAdapter
from builders.base import BaseBuilder


class TemporalGroundingBuilder(BaseBuilder):
    capability = SubCapability.TEMPORAL_GROUNDING
    capability_name = "c_temporal_grounding"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            QVHighlightsAdapter(data_root=self.data_root),
            ActivityNetCaptionsAdapter(data_root=self.data_root),
        ]

    def build_samples(self, adapter: BaseAdapter, split: str) -> list[CanonicalSample]:
        samples = []
        for raw in adapter.iterate_raw(split):
            sample = adapter.to_canonical(
                raw,
                sub_capability=self.capability,
                task_type=TaskType.GROUNDING,
            )
            samples.append(sample)
        return samples
