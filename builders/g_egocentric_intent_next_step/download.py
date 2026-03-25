"""Download script generator for Egocentric Intent & Next Step capability."""

from builders.g_egocentric_intent_next_step.build import EgocentricIntentNextStepBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Egocentric Intent & Next Step adapters."""
    builder = EgocentricIntentNextStepBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
