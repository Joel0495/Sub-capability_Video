"""Stats checker: count samples per capability and check against success criteria."""

import json
from pathlib import Path

import fire


# Success criteria
TARGET_CANONICAL = 200_000
TARGET_RL_READY = 80_000
MIN_SOURCES_PER_CAPABILITY = 3


def _count_jsonl(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    for line in path.open():
        if line.strip():
            count += 1
    return count


def _count_sources(path: Path) -> set:
    """Collect unique datasource values from a JSONL file."""
    sources = set()
    if not path.exists():
        return sources
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            ds = sample.get("data_info", {}).get("datasource", "")
            if ds:
                sources.add(ds)
        except json.JSONDecodeError:
            pass
    return sources


def stats(data_root: str = "data"):
    """Count total samples per capability and check success criteria.

    Args:
        data_root: Root data directory.
    """
    root = Path(data_root)
    canonical_dir = root / "canonical"
    rl_ready_dir = root / "rl_ready"

    # Per-capability stats
    canonical_total = 0
    rl_ready_total = 0
    capability_stats = []
    source_issues = []

    if canonical_dir.exists():
        for jsonl_file in sorted(canonical_dir.glob("*.jsonl")):
            count = _count_jsonl(jsonl_file)
            sources = _count_sources(jsonl_file)
            canonical_total += count
            capability_stats.append({
                "file": jsonl_file.name,
                "canonical_count": count,
                "sources": sources,
            })
            if len(sources) < MIN_SOURCES_PER_CAPABILITY:
                source_issues.append(
                    f"{jsonl_file.name}: only {len(sources)} sources "
                    f"(need >= {MIN_SOURCES_PER_CAPABILITY}): {sorted(sources)}"
                )

    if rl_ready_dir.exists():
        for jsonl_file in sorted(rl_ready_dir.glob("*.jsonl")):
            count = _count_jsonl(jsonl_file)
            rl_ready_total += count
            # Update matching capability stat if it exists
            for cs in capability_stats:
                if cs["file"] == jsonl_file.name:
                    cs["rl_ready_count"] = count
                    break

    # Print report
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"{'Capability':<45} {'Canonical':>10} {'Sources':>8}")
    print("-" * 70)
    for cs in capability_stats:
        src_count = len(cs["sources"])
        print(f"  {cs['file']:<43} {cs['canonical_count']:>10} {src_count:>8}")
    print("-" * 70)
    print(f"  {'TOTAL CANONICAL':<43} {canonical_total:>10}")
    print(f"  {'TOTAL RL-READY':<43} {rl_ready_total:>10}")

    # Check criteria
    print(f"\n{'=' * 70}")
    print("SUCCESS CRITERIA")
    print("=" * 70)

    pass_canonical = canonical_total >= TARGET_CANONICAL
    pass_rl_ready = rl_ready_total >= TARGET_RL_READY
    pass_sources = len(source_issues) == 0

    status_c = "PASS" if pass_canonical else "FAIL"
    status_r = "PASS" if pass_rl_ready else "FAIL"
    status_s = "PASS" if pass_sources else "FAIL"

    print(f"  [{status_c}] Canonical samples: {canonical_total:,} / {TARGET_CANONICAL:,}")
    print(f"  [{status_r}] RL-ready samples:  {rl_ready_total:,} / {TARGET_RL_READY:,}")
    print(f"  [{status_s}] Min {MIN_SOURCES_PER_CAPABILITY} sources per capability")

    if source_issues:
        for issue in source_issues:
            print(f"         {issue}")

    overall = pass_canonical and pass_rl_ready and pass_sources
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    fire.Fire(stats)
