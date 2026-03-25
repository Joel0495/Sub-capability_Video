"""Download script generator for Long Video Retrieval & Memory capability."""

from builders.d_long_video_retrieval_memory.build import LongVideoRetrievalMemoryBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Long Video Retrieval & Memory adapters."""
    builder = LongVideoRetrievalMemoryBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
