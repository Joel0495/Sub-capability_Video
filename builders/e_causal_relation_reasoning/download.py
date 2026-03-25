"""Download script generator for Causal & Relation Reasoning capability."""

from builders.e_causal_relation_reasoning.build import CausalRelationReasoningBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Causal & Relation Reasoning adapters."""
    builder = CausalRelationReasoningBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
