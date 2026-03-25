"""Builder for sub-capability J: Topic & Plot Knowledge Acquisition.

Uses TVQAAdapter and ActivityNetCaptionsAdapter for topic and plot
knowledge data.
"""

from schema.canonical import CanonicalSample, SubCapability, TaskType
from source_adapters.base_adapter import BaseAdapter
from source_adapters.tvqa import TVQAAdapter
from source_adapters.activitynet_captions import ActivityNetCaptionsAdapter
from builders.base import BaseBuilder


class TopicPlotKnowledgeBuilder(BaseBuilder):
    capability = SubCapability.TOPIC_PLOT_KNOWLEDGE_ACQUISITION
    capability_name = "j_topic_plot_knowledge"

    def get_adapters(self) -> list[BaseAdapter]:
        return [
            TVQAAdapter(data_root=self.data_root),
            ActivityNetCaptionsAdapter(data_root=self.data_root),
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
