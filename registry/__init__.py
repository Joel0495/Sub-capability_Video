"""Benchmark hold-out registry and dataset safety checks."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional

_HOLDOUT = None
_HOLDOUT_PATH = Path(__file__).parent / "benchmark_holdout.yaml"


def load_benchmark_holdout(path: Optional[Path] = None) -> dict:
    """Load the benchmark holdout registry."""
    global _HOLDOUT
    if _HOLDOUT is None or path is not None:
        p = path or _HOLDOUT_PATH
        with open(p) as f:
            _HOLDOUT = yaml.safe_load(f)
    return _HOLDOUT


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def is_benchmark(dataset_name: str, holdout: Optional[dict] = None) -> bool:
    """Check if a dataset is a pure evaluation benchmark (forbidden for training)."""
    holdout = holdout or load_benchmark_holdout()
    benchmark_names = {_normalize(b["name"]) for b in holdout["benchmarks"]}
    return _normalize(dataset_name) in benchmark_names


def get_allowed_splits(dataset_name: str, holdout: Optional[dict] = None) -> Optional[list[str]]:
    """Get allowed splits for a dataset, or None if unrestricted."""
    holdout = holdout or load_benchmark_holdout()
    for entry in holdout.get("split_restricted", []):
        if _normalize(entry["name"]) == _normalize(dataset_name):
            return entry["allowed_splits"]
    return None


def assert_not_benchmark(dataset_name: str, holdout: Optional[dict] = None) -> None:
    """Raise if dataset is a benchmark."""
    if is_benchmark(dataset_name, holdout):
        raise ValueError(
            f"BLOCKED: '{dataset_name}' is a pure evaluation benchmark. "
            f"Using it for training would contaminate evaluation results."
        )


def assert_valid_split(dataset_name: str, split: str, holdout: Optional[dict] = None) -> None:
    """Raise if using a forbidden split (e.g., test split)."""
    allowed = get_allowed_splits(dataset_name, holdout)
    if allowed is not None and split not in allowed:
        raise ValueError(
            f"Split '{split}' of '{dataset_name}' is forbidden for training. "
            f"Allowed splits: {allowed}"
        )
