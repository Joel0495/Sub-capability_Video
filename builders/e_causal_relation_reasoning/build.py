"""Builder for sub-capability E: Causal & Relation Reasoning.

Uses NExTQAAdapter, STARAdapter, and CLEVRERAdapter for causal reasoning data.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.nextqa import NExTQAAdapter
from source_adapters.star import STARAdapter
from source_adapters.clevrer import CLEVRERAdapter
from builders.base import BaseBuilder


class CausalRelationReasoningBuilder(BaseBuilder):
    capability = SubCapability.CAUSAL_RELATION_REASONING
    capability_name = "e_causal_relation_reasoning"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            NExTQAAdapter(data_root=self.data_root),
            STARAdapter(data_root=self.data_root),
            CLEVRERAdapter(data_root=self.data_root),
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
