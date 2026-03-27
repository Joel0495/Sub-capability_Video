"""Builder for sub-capability H: Spatial & Spatiotemporal Reasoning.

Uses SpaceRAdapter and CLEVRERAdapter for spatial and spatiotemporal
reasoning data.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.spacer import SpaceRAdapter
from source_adapters.clevrer import CLEVRERAdapter
from builders.base import BaseBuilder


class SpatialSpatiotemporalBuilder(BaseBuilder):
    capability = SubCapability.SPATIAL_SPATIOTEMPORAL_REASONING
    capability_name = "h_spatial_spatiotemporal"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            SpaceRAdapter(data_root=self.data_root),
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
