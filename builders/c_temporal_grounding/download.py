"""Download script generator for Temporal Grounding capability."""

from builders.c_temporal_grounding.build import TemporalGroundingBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Temporal Grounding adapters."""
    builder = TemporalGroundingBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
