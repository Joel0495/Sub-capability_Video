"""Download script generator for Temporal Atomic capability."""

from builders.a_temporal_atomic.build import TemporalAtomicBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Temporal Atomic adapters."""
    builder = TemporalAtomicBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
