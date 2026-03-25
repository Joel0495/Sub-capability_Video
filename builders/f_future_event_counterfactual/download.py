"""Download script generator for Future Event & Counterfactual capability."""

from builders.f_future_event_counterfactual.build import FutureEventCounterfactualBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Future Event & Counterfactual adapters."""
    builder = FutureEventCounterfactualBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
