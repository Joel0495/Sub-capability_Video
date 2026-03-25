"""Base builder class for all capability builders.

Each builder:
1. Checks that none of its source datasets are benchmarks
2. Iterates source adapters' train splits
3. Converts to canonical schema
4. Writes JSONL output
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from schema.canonical import CanonicalSample, SubCapability
from source_adapters.base_adapter import BaseAdapter
from registry import assert_not_benchmark, assert_valid_split


class BaseBuilder(ABC):
    """Abstract builder for one sub-capability."""

    capability: SubCapability
    capability_name: str = ""  # e.g., "a_temporal_atomic"

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.canonical_dir = self.data_root / "canonical"
        self.rl_ready_dir = self.data_root / "rl_ready"
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.rl_ready_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_adapters(self) -> list[BaseAdapter]:
        """Return the list of source adapters for this capability."""
        ...

    def validate_sources(self) -> None:
        """Ensure no adapter references a benchmark dataset."""
        for adapter in self.get_adapters():
            assert_not_benchmark(adapter.dataset_name)

    def build(self, split: str = "train") -> dict:
        """Build canonical JSONL for this capability.

        Returns:
            stats dict with counts
        """
        self.validate_sources()

        out_path = self.canonical_dir / f"{self.capability_name}.jsonl"
        count = 0
        source_counts: dict[str, int] = {}

        with open(out_path, "w") as f:
            for adapter in self.get_adapters():
                assert_valid_split(adapter.dataset_name, split)
                adapter_count = 0

                for sample in self.build_samples(adapter, split):
                    f.write(sample.model_dump_json() + "\n")
                    count += 1
                    adapter_count += 1

                source_counts[adapter.dataset_name] = adapter_count

        return {
            "capability": self.capability_name,
            "total": count,
            "output": str(out_path),
            "sources": source_counts,
        }

    @abstractmethod
    def build_samples(
        self, adapter: BaseAdapter, split: str
    ) -> list[CanonicalSample]:
        """Build canonical samples from one adapter's data.

        Override in each capability builder to apply capability-specific
        filtering, question formatting, and grader assignment.
        """
        ...

    def generate_download_script(self) -> str:
        """Generate the download script for all adapters in this builder."""
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"# Download script for {self.capability_name}",
            "",
        ]
        seen = set()
        for adapter in self.get_adapters():
            if adapter.dataset_name not in seen:
                lines.append(adapter.generate_download_script())
                seen.add(adapter.dataset_name)
        return "\n".join(lines)
