"""Download script generator for Temporal Count & Order capability."""

from builders.b_temporal_count_order.build import TemporalCountOrderBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Temporal Count & Order adapters."""
    builder = TemporalCountOrderBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
