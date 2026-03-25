"""Merger: combine all canonical and rl_ready JSONL files into merged outputs."""

import json
from pathlib import Path

import fire


def _merge_jsonl_dir(src_dir: Path, output_path: Path) -> int:
    """Merge all JSONL files from src_dir into a single output file.

    Returns the total number of samples written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    count = 0
    with open(output_path, "w") as out:
        for jsonl_file in jsonl_files:
            for line in jsonl_file.open():
                line = line.strip()
                if not line:
                    continue
                out.write(line + "\n")
                count += 1
    return count


def merge(
    data_root: str = "data",
):
    """Merge all canonical and rl_ready JSONL files into merged outputs.

    Args:
        data_root: Root data directory.
    """
    root = Path(data_root)
    canonical_dir = root / "canonical"
    rl_ready_dir = root / "rl_ready"
    merged_dir = root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Merge canonical
    canonical_out = merged_dir / "canonical_all.jsonl"
    if canonical_dir.exists():
        canonical_count = _merge_jsonl_dir(canonical_dir, canonical_out)
        print(f"Canonical: {canonical_count} samples -> {canonical_out}")
    else:
        canonical_count = 0
        print(f"Canonical directory not found: {canonical_dir}")

    # Merge rl_ready
    rl_ready_out = merged_dir / "rl_ready_all.jsonl"
    if rl_ready_dir.exists():
        rl_count = _merge_jsonl_dir(rl_ready_dir, rl_ready_out)
        print(f"RL-ready:  {rl_count} samples -> {rl_ready_out}")
    else:
        rl_count = 0
        print(f"RL-ready directory not found: {rl_ready_dir}")

    print(f"\nMerge complete: {canonical_count} canonical, {rl_count} rl_ready")


if __name__ == "__main__":
    fire.Fire(merge)
