"""JSONL validation utilities for canonical samples."""

import json
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from schema.canonical import CanonicalSample


def validate_jsonl(path: str | Path) -> tuple[int, int, list[str]]:
    """Validate a JSONL file against the canonical schema.

    Returns:
        (valid_count, error_count, error_messages)
    """
    path = Path(path)
    valid = 0
    errors = []
    for i, line in enumerate(path.open(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            CanonicalSample.model_validate(data)
            valid += 1
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(f"Line {i}: {e}")
    return valid, len(errors), errors


def iterate_validated(path: str | Path) -> Iterator[CanonicalSample]:
    """Iterate over a JSONL file, yielding validated CanonicalSample objects."""
    path = Path(path)
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        yield CanonicalSample.model_validate(data)


def check_video_files_exist(path: str | Path) -> list[str]:
    """Check that all video_url references point to existing files."""
    missing = []
    for sample in iterate_validated(path):
        for msg in sample.messages:
            for content in msg.content:
                if hasattr(content, "video_url"):
                    url = content.video_url.url
                    if url.startswith("file://"):
                        file_path = url[7:]  # strip file://
                        if not Path(file_path).exists():
                            missing.append(f"{sample.data_info.data_id}: {url}")
    return missing
