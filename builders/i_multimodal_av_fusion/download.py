"""Download script generator for Multimodal AV Fusion capability."""

from builders.i_multimodal_av_fusion.build import MultimodalAVFusionBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Multimodal AV Fusion adapters."""
    builder = MultimodalAVFusionBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
