"""Builder for sub-capability F: Future Event & Counterfactual.

Uses NEPV1Adapter, STARAdapter, CLEVRERAdapter, and Ego4DAdapter for
future prediction and counterfactual reasoning data.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.nep_v1 import NEPV1Adapter
from source_adapters.star import STARAdapter
from source_adapters.clevrer import CLEVRERAdapter
from source_adapters.ego4d import Ego4DAdapter
from builders.base import BaseBuilder


class FutureEventCounterfactualBuilder(BaseBuilder):
    capability = SubCapability.FUTURE_EVENT_COUNTERFACTUAL
    capability_name = "f_future_event_counterfactual"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            NEPV1Adapter(data_root=self.data_root),
            STARAdapter(data_root=self.data_root),
            CLEVRERAdapter(data_root=self.data_root),
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
