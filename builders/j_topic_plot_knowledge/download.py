"""Download script generator for Topic & Plot Knowledge Acquisition capability."""

from builders.j_topic_plot_knowledge.build import TopicPlotKnowledgeBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Topic & Plot Knowledge adapters."""
    builder = TopicPlotKnowledgeBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
