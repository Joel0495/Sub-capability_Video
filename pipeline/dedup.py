"""Deduplication: remove duplicate samples by (video_url hash + normalized question)."""

import hashlib
import json
from pathlib import Path

import fire


def _normalize_question(text: str) -> str:
    """Lowercase, strip whitespace, collapse spaces."""
    return " ".join(text.lower().split())


def _extract_video_url(sample: dict) -> str:
    """Extract the first video_url from a sample's messages."""
    for msg in sample.get("messages", []):
        for content in msg.get("content", []):
            if content.get("type") == "video_url":
                return content.get("video_url", {}).get("url", "")
    return ""


def _extract_question(sample: dict) -> str:
    """Extract the question text from the user message."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            for content in msg.get("content", []):
                if content.get("type") == "text":
                    return content.get("text", "")
    return ""


def _dedup_key(sample: dict) -> str:
    """Create a dedup key from video URL hash + normalized question."""
    video_url = _extract_video_url(sample)
    question = _normalize_question(_extract_question(sample))
    url_hash = hashlib.sha256(video_url.encode()).hexdigest()[:16]
    q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    return f"{url_hash}_{q_hash}"


def dedup(canonical_dir: str = "data/canonical", dry_run: bool = False):
    """Deduplicate all canonical JSONL files in place.

    Args:
        canonical_dir: Directory containing canonical JSONL files.
        dry_run: If True, print stats but do not write files.
    """
    canonical_path = Path(canonical_dir)
    if not canonical_path.exists():
        print(f"Directory not found: {canonical_path}")
        return

    jsonl_files = sorted(canonical_path.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found.")
        return

    global_seen = set()
    total_before = 0
    total_after = 0

    print(f"Deduplicating {len(jsonl_files)} JSONL files in {canonical_path}\n")

    for jsonl_file in jsonl_files:
        samples = []
        kept = []
        for line in jsonl_file.open():
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            samples.append(sample)
            key = _dedup_key(sample)
            if key not in global_seen:
                global_seen.add(key)
                kept.append(sample)

        removed = len(samples) - len(kept)
        total_before += len(samples)
        total_after += len(kept)

        status = f"  {jsonl_file.name}: {len(samples)} -> {len(kept)} ({removed} removed)"
        print(status)

        if not dry_run and removed > 0:
            with open(jsonl_file, "w") as f:
                for sample in kept:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nTotal: {total_before} -> {total_after} ({total_before - total_after} duplicates removed)")
    if dry_run:
        print("(dry run - no files modified)")


if __name__ == "__main__":
    fire.Fire(dedup)
