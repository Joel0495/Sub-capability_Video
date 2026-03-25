"""Builder for sub-capability K: Anti-Shortcut & Contrast.

Uses VideoR1Adapter, SpaceRAdapter, and CLEVRERAdapter for anti-shortcut
and contrastive reasoning data.
"""

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.video_r1 import VideoR1Adapter
from source_adapters.spacer import SpaceRAdapter
from source_adapters.clevrer import CLEVRERAdapter
from builders.base import BaseBuilder


class AntiShortcutContrastBuilder(BaseBuilder):
    capability = SubCapability.ANTI_SHORTCUT_CONTRAST
    capability_name = "k_anti_shortcut_contrast"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            VideoR1Adapter(data_root=self.data_root),
            SpaceRAdapter(data_root=self.data_root),
            CLEVRERAdapter(data_root=self.data_root),
        ]

    def build_samples(self, adapter: BaseAdapter, split: str) -> list[CanonicalSample]:
        samples = []
        for raw in adapter.iterate_raw(split):
            sample = adapter.to_canonical(
                raw,
                sub_capability=self.capability,
                task_type=TaskType.PREFERENCE_PAIR,
            )
            samples.append(sample)
        return samples
