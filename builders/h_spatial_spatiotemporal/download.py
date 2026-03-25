"""Download script generator for Spatial & Spatiotemporal Reasoning capability."""

from builders.h_spatial_spatiotemporal.build import SpatialSpatiotemporalBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Spatial & Spatiotemporal adapters."""
    builder = SpatialSpatiotemporalBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
