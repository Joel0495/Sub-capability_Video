"""Analyze diagnostic evaluation results per sub-capability.

Usage:
    python -m scripts.analyze_diagnostic --predictions predictions.jsonl

Input format: each line is a JSON with at least:
    {"data_id": "diag_ta_001", "prediction": "B"}

Or you can pass --gt_file to compare against the diagnostic JSONL directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from graders.ruler_graders import exact_match, option_match


def load_gt(path: str = "data/diagnostic/diagnostic_eval.jsonl") -> dict:
    """Load ground truth from diagnostic JSONL.

    Returns: {data_id: {"answer": str, "bucket": str, "sub_ability": list, "task_type": str}}
    """
    gt = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            data_id = sample["data_info"]["data_id"]
            # Extract answer from assistant message
            answer = ""
            for msg in sample["messages"]:
                if msg["role"] == "assistant":
                    for c in msg["content"]:
                        if c.get("type") == "text":
                            answer = c["text"]
            gt[data_id] = {
                "answer": answer,
                "bucket": sample["extra_info"]["sampling_info"]["mix_bucket"],
                "sub_ability": sample["data_info"]["sub_ability"],
                "task_type": sample["data_info"]["task_type"],
                "grader": sample["graders"][0]["name"],
            }
    return gt


def evaluate(
    predictions_path: str,
    gt_path: str = "data/diagnostic/diagnostic_eval.jsonl",
):
    """Evaluate predictions against ground truth, report per-capability accuracy."""
    gt = load_gt(gt_path)
    predictions = {}
    with open(predictions_path) as f:
        for line in f:
            if not line.strip():
                continue
            pred = json.loads(line)
            data_id = pred.get("data_id", pred.get("id", ""))
            prediction = pred.get("prediction", pred.get("response", pred.get("answer", "")))
            predictions[data_id] = prediction

    # Score each sample
    results_by_cap = defaultdict(lambda: {"correct": 0, "total": 0, "details": []})
    results_by_sub = defaultdict(lambda: {"correct": 0, "total": 0})

    for data_id, info in gt.items():
        pred = predictions.get(data_id, "")
        gt_ans = info["answer"]
        bucket = info["bucket"]
        grader_name = info["grader"]

        # Grade
        if grader_name == "option_match":
            score = option_match(pred, gt_ans)
        elif grader_name == "exact_match":
            score = exact_match(pred, gt_ans)
        else:
            score = exact_match(pred, gt_ans)  # fallback

        correct = score >= 1.0
        results_by_cap[bucket]["total"] += 1
        results_by_cap[bucket]["correct"] += int(correct)
        results_by_cap[bucket]["details"].append({
            "id": data_id,
            "pred": pred,
            "gt": gt_ans,
            "correct": correct,
        })

        for sub in info["sub_ability"]:
            results_by_sub[sub]["total"] += 1
            results_by_sub[sub]["correct"] += int(correct)

    # Print report
    total_correct = sum(v["correct"] for v in results_by_cap.values())
    total_all = sum(v["total"] for v in results_by_cap.values())

    print("=" * 70)
    print(f"  DIAGNOSTIC EVALUATION REPORT  ({total_correct}/{total_all} = {total_correct/max(total_all,1)*100:.1f}%)")
    print("=" * 70)

    # Sorted by accuracy (worst first = weakest capabilities first)
    cap_scores = []
    for cap in sorted(results_by_cap.keys()):
        r = results_by_cap[cap]
        acc = r["correct"] / max(r["total"], 1)
        cap_scores.append((cap, acc, r["correct"], r["total"]))

    cap_scores.sort(key=lambda x: x[1])  # worst first

    print("\n  Sub-capability (sorted by accuracy, weakest first):")
    print("  " + "-" * 56)
    for cap, acc, corr, tot in cap_scores:
        bar = "#" * int(acc * 20) + "." * (20 - int(acc * 20))
        status = "WEAK" if acc < 0.6 else ("OK" if acc < 0.8 else "GOOD")
        print(f"  {cap:<42s} {corr}/{tot}  {acc*100:5.1f}%  [{bar}] {status}")

    # Sub-ability breakdown
    print("\n  Sub-ability detail:")
    print("  " + "-" * 56)
    sub_scores = []
    for sub, r in results_by_sub.items():
        acc = r["correct"] / max(r["total"], 1)
        sub_scores.append((sub, acc, r["correct"], r["total"]))
    sub_scores.sort(key=lambda x: x[1])
    for sub, acc, corr, tot in sub_scores:
        print(f"  {sub:<42s} {corr}/{tot}  {acc*100:5.1f}%")

    # List wrong answers
    print("\n  Incorrect predictions:")
    print("  " + "-" * 56)
    wrong_count = 0
    for cap in sorted(results_by_cap.keys()):
        for d in results_by_cap[cap]["details"]:
            if not d["correct"]:
                wrong_count += 1
                print(f"  [{cap}] {d['id']}: pred='{d['pred']}' gt='{d['gt']}'")
    if wrong_count == 0:
        print("  (none - all correct!)")

    print("\n" + "=" * 70)
    return {cap: acc for cap, acc, _, _ in cap_scores}


if __name__ == "__main__":
    import fire
    fire.Fire(evaluate)
