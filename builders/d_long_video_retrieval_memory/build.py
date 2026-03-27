"""Builder for sub-capability D: Long Video Retrieval & Memory.

Uses TVQAAdapter, Ego4DAdapter, and ActivityNetCaptionsAdapter for long
video memory and retrieval data.
"""

from __future__ import annotations

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.tvqa import TVQAAdapter
from source_adapters.ego4d import Ego4DAdapter
from source_adapters.activitynet_captions import ActivityNetCaptionsAdapter
from builders.base import BaseBuilder


class LongVideoRetrievalMemoryBuilder(BaseBuilder):
    capability = SubCapability.LONG_VIDEO_RETRIEVAL_MEMORY
    capability_name = "d_long_video_retrieval_memory"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            TVQAAdapter(data_root=self.data_root),
            Ego4DAdapter(data_root=self.data_root),
            ActivityNetCaptionsAdapter(data_root=self.data_root),
        ]

    def build_samples(self, adapter: BaseAdapter, split: str) -> list[CanonicalSample]:
        samples = []
        for raw in adapter.iterate_raw(split):
            sample = adapter.to_canonical(
                raw,
                sub_capability=self.capability,
                task_type=TaskType.RETRIEVAL,
            )
            samples.append(sample)
        return samples
