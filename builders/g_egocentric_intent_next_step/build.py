"""Builder for sub-capability G: Egocentric Intent & Next Step.

Uses Ego4DAdapter for egocentric video understanding data.
"""

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.ego4d import Ego4DAdapter
from builders.base import BaseBuilder


class EgocentricIntentNextStepBuilder(BaseBuilder):
    capability = SubCapability.EGOCENTRIC_INTENT_NEXT_STEP
    capability_name = "g_egocentric_intent_next_step"

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
