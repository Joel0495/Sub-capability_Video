"""Orchestrator: collect all datasets across builders and generate one master download script."""

from __future__ import annotations

import yaml
from pathlib import Path

import fire

from builders.base import BaseBuilder


def _load_download_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "download_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_capability_map() -> dict:
    cap_path = Path(__file__).parent.parent / "registry" / "capability_map.yaml"
    with open(cap_path) as f:
        return yaml.safe_load(f)


def _collect_datasets(priority_only: bool = False) -> list[str]:
    """Collect all unique dataset names across all capabilities, deduped."""
    cap_map = _load_capability_map()
    seen = set()
    ordered = []
    for cap_name, cap_info in cap_map["capabilities"].items():
        sources = cap_info.get("sources", {})
        priorities = ["priority_1"]
        if not priority_only:
            priorities += ["priority_2", "priority_3"]
        for prio in priorities:
            for ds in sources.get(prio, []) or []:
                if ds not in seen:
                    seen.add(ds)
                    ordered.append(ds)
    return ordered


def generate(priority: bool = False, output: str = "data/download_all.sh"):
    """Generate a master download script for all datasets.

    Args:
        priority: If True, only include priority-1 datasets.
        output: Path for the generated shell script.
    """
    config = _load_download_config()
    dl = config["download"]
    hf_dl = dl["hf_dl_script"]
    browser = dl["browser_executable_path"]
    concurrency = dl["scrape_concurrency"]
    data_root = dl["data_root"]

    hf_datasets = config.get("hf_datasets", {})
    manual_datasets = config.get("manual_datasets", {})

    datasets = _collect_datasets(priority_only=priority)

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "# ============================================",
        "# Master download script (auto-generated)",
        f"# Priority-only: {priority}",
        f"# Datasets: {len(datasets)}",
        "# ============================================",
        "",
    ]

    hf_count = 0
    manual_count = 0

    for ds_name in datasets:
        if ds_name in hf_datasets:
            info = hf_datasets[ds_name]
            repo_id = info["repo_id"]
            ds_type = info.get("type", "dataset")
            note = info.get("note", "")
            raw_dir = f"{data_root}/{ds_name}"
            lines.append(f"# === {ds_name} ===")
            lines.append(f"# {note}")
            lines.append(f"echo 'Downloading {ds_name} from {repo_id}...'")
            lines.append(f'bash {hf_dl} {repo_id} {ds_type} \\')
            lines.append(f'    --browser-executable-path "{browser}" \\')
            lines.append(f'    --scrape-concurrency {concurrency}')
            lines.append(f"mkdir -p {raw_dir}/videos {raw_dir}/annotations")
            lines.append("")
            hf_count += 1
        elif ds_name in manual_datasets:
            info = manual_datasets[ds_name]
            instructions = info.get("instructions", "See source URL.")
            raw_dir = f"{data_root}/{ds_name}"
            lines.append(f"# === {ds_name} (MANUAL) ===")
            lines.append(f"# MANUAL DOWNLOAD REQUIRED")
            for inst_line in instructions.strip().splitlines():
                lines.append(f"# {inst_line}")
            lines.append(f"mkdir -p {raw_dir}/videos {raw_dir}/annotations")
            lines.append("")
            manual_count += 1
        else:
            lines.append(f"# === {ds_name} ===")
            lines.append(f"# WARNING: No download config found for '{ds_name}'")
            lines.append(f"# Please add to configs/download_config.yaml")
            lines.append("")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

    print(f"Generated {out_path}")
    print(f"  HuggingFace datasets: {hf_count}")
    print(f"  Manual datasets:      {manual_count}")
    print(f"  Total unique datasets: {len(datasets)}")


if __name__ == "__main__":
    fire.Fire(generate)
