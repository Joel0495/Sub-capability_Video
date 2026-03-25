"""Download script generator for Anti-Shortcut & Contrast capability."""

from builders.k_anti_shortcut_contrast.build import AntiShortcutContrastBuilder


def main(data_root: str = "data") -> str:
    """Generate the download script for Anti-Shortcut & Contrast adapters."""
    builder = AntiShortcutContrastBuilder(data_root=data_root)
    return builder.generate_download_script()


if __name__ == "__main__":
    print(main())
