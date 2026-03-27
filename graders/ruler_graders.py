"""Rule-based grader implementations.

Prefer ruler graders whenever possible. Model graders are fallback only.
"""

from __future__ import annotations

import re
from typing import Optional


def exact_match(prediction: str, gt: str) -> float:
    """Exact string match after normalization."""
    pred = prediction.strip().lower()
    truth = gt.strip().lower()
    return 1.0 if pred == truth else 0.0


def option_match(prediction: str, gt: str) -> float:
    """Match a single capital letter option (A/B/C/D).

    Tries multiple extraction strategies:
    1. If prediction is a single letter, use it directly
    2. Look for standalone capital letter (word boundary)
    3. Fall back to last capital letter found
    """
    pred = prediction.strip()
    truth = gt.strip().upper()

    # Single letter answer
    if len(pred) == 1:
        return 1.0 if pred.upper() == truth else 0.0

    # Look for standalone capital letter (e.g., "The answer is B" -> "B")
    standalone = re.findall(r"\b([A-Z])\b", pred)
    if standalone:
        return 1.0 if standalone[-1] == truth else 0.0

    # Fall back to last capital letter in A-H range
    matches = re.findall(r"[A-H]", pred.upper())
    if matches:
        return 1.0 if matches[-1] == truth else 0.0

    return 0.0


def regex_extract_then_match(prediction: str, gt: str, pattern: str) -> float:
    """Extract answer via regex, then exact match."""
    match = re.search(pattern, prediction, re.IGNORECASE)
    if match:
        extracted = match.group(1) if match.groups() else match.group()
        return exact_match(extracted, gt)
    return 0.0


def count_abs_error(prediction: str, gt: str, tolerance: int = 0) -> float:
    """Score based on absolute count error."""
    try:
        pred_num = int(re.search(r"\d+", prediction).group())
        gt_num = int(gt)
        error = abs(pred_num - gt_num)
        if error <= tolerance:
            return 1.0
        return max(0.0, 1.0 - error / max(gt_num, 1))
    except (AttributeError, ValueError):
        return 0.0


def count_rel_error(prediction: str, gt: str, threshold: float = 0.1) -> float:
    """Score based on relative count error."""
    try:
        pred_num = float(re.search(r"[\d.]+", prediction).group())
        gt_num = float(gt)
        if gt_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        rel_error = abs(pred_num - gt_num) / abs(gt_num)
        return 1.0 if rel_error <= threshold else max(0.0, 1.0 - rel_error)
    except (AttributeError, ValueError):
        return 0.0


def order_match(prediction: str, gt: str) -> float:
    """Match an ordered sequence (e.g., 'A,B,C' vs 'A,B,C')."""
    pred_items = [x.strip().upper() for x in re.split(r"[,;\s]+", prediction.strip())]
    gt_items = [x.strip().upper() for x in re.split(r"[,;\s]+", gt.strip())]
    if pred_items == gt_items:
        return 1.0
    # Partial credit: fraction of items in correct position
    if len(pred_items) != len(gt_items):
        return 0.0
    correct = sum(1 for p, g in zip(pred_items, gt_items) if p == g)
    return correct / len(gt_items)


def span_iou(
    pred_spans: list[list[float]],
    gt_spans: list[list[float]],
) -> float:
    """Compute IoU between predicted and ground-truth temporal spans."""
    if not pred_spans or not gt_spans:
        return 0.0

    def _iou(a: list[float], b: list[float]) -> float:
        inter_start = max(a[0], b[0])
        inter_end = min(a[1], b[1])
        inter = max(0.0, inter_end - inter_start)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0.0

    # Best matching: for each gt span, find best pred span IoU
    total_iou = 0.0
    for gs in gt_spans:
        best = max(_iou(ps, gs) for ps in pred_spans)
        total_iou += best
    return total_iou / len(gt_spans)


def retrieval_hit_at_k(
    prediction: str,
    gt: str,
    k: int = 5,
) -> float:
    """Check if ground truth appears in top-k predicted items."""
    pred_items = [x.strip().lower() for x in prediction.split(",")][:k]
    return 1.0 if gt.strip().lower() in pred_items else 0.0


def pairwise_pref(
    score_chosen: float,
    score_rejected: float,
) -> float:
    """Preference grader: 1.0 if chosen > rejected, 0.0 otherwise."""
    return 1.0 if score_chosen > score_rejected else 0.0


def format_checker(prediction: str, expected_format: str = "letter") -> float:
    """Check if prediction follows the expected format."""
    prediction = prediction.strip()
    if expected_format == "letter":
        return 1.0 if re.match(r"^[A-Z]$", prediction) else 0.0
    elif expected_format == "number":
        return 1.0 if re.match(r"^\d+$", prediction) else 0.0
    elif expected_format == "span":
        return 1.0 if re.match(r"^\d+\.?\d*\s*[-,]\s*\d+\.?\d*$", prediction) else 0.0
    return 1.0  # no format check


def keyword_set_match(
    prediction: str,
    keywords: list[str],
    require_all: bool = False,
) -> float:
    """Check if prediction contains required keywords."""
    pred_lower = prediction.lower()
    matches = sum(1 for kw in keywords if kw.lower() in pred_lower)
    if require_all:
        return 1.0 if matches == len(keywords) else 0.0
    return matches / len(keywords) if keywords else 0.0
