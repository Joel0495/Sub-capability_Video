"""Validator: validate all JSONL files against the canonical schema and check for issues."""

import json
from pathlib import Path

import fire

from schema.validators import validate_jsonl, check_video_files_exist
from registry import is_benchmark


def _check_benchmark_leakage(jsonl_path: Path) -> list[str]:
    """Check that no samples reference a benchmark dataset."""
    issues = []
    for i, line in enumerate(jsonl_path.open(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            datasource = sample.get("data_info", {}).get("datasource", "")
            if datasource and is_benchmark(datasource):
                issues.append(
                    f"Line {i}: benchmark dataset '{datasource}' found in {jsonl_path.name}"
                )
        except json.JSONDecodeError:
            pass
    return issues


def validate(
    data_root: str = "data",
    check_videos: bool = False,
):
    """Validate all JSONL files in canonical/ and rl_ready/ directories.

    Args:
        data_root: Root data directory.
        check_videos: If True, also check that referenced video files exist.
    """
    root = Path(data_root)
    dirs_to_check = []
    for subdir in ["canonical", "rl_ready", "merged"]:
        d = root / subdir
        if d.exists():
            dirs_to_check.append(d)

    if not dirs_to_check:
        print("No data directories found to validate.")
        return

    total_valid = 0
    total_errors = 0
    all_issues = []

    for data_dir in dirs_to_check:
        jsonl_files = sorted(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue

        print(f"\nValidating {data_dir}/")
        for jsonl_file in jsonl_files:
            valid, error_count, errors = validate_jsonl(jsonl_file)
            total_valid += valid
            total_errors += error_count

            # Check benchmark leakage
            leakage = _check_benchmark_leakage(jsonl_file)
            all_issues.extend(leakage)

            # Check video files
            if check_videos:
                missing = check_video_files_exist(jsonl_file)
                all_issues.extend([f"Missing video: {m}" for m in missing])

            status = "OK" if error_count == 0 and not leakage else "FAIL"
            print(f"  [{status}] {jsonl_file.name}: {valid} valid, {error_count} errors")
            if errors:
                for err in errors[:5]:
                    print(f"         {err}")
                if len(errors) > 5:
                    print(f"         ... and {len(errors) - 5} more errors")
            if leakage:
                for issue in leakage:
                    print(f"         BENCHMARK LEAK: {issue}")

    print(f"\n{'=' * 60}")
    print(f"Validation complete: {total_valid} valid, {total_errors} errors, {len(all_issues)} issues")
    if all_issues:
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")


if __name__ == "__main__":
    fire.Fire(validate)
