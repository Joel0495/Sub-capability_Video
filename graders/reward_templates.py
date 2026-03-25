"""Unified reward templates for the 11 sub-capabilities.

Each template computes a weighted combination of component scores.
"""

import yaml
from pathlib import Path
from typing import Optional


def _load_templates() -> dict:
    path = Path(__file__).parent.parent / "configs" / "reward_templates.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["templates"]


_TEMPLATES: Optional[dict] = None


def get_template(name: str) -> dict:
    """Get a reward template by name."""
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = _load_templates()
    if name not in _TEMPLATES:
        raise KeyError(f"Unknown reward template: {name}. Available: {list(_TEMPLATES.keys())}")
    return _TEMPLATES[name]


def compute_reward(template_name: str, scores: dict[str, float]) -> float:
    """Compute weighted reward using a named template.

    Args:
        template_name: e.g., "temporal_atomic_v1"
        scores: dict of component scores, e.g., {"ans": 1.0, "temporal": 0.5, "format": 1.0}

    Returns:
        Weighted reward in [0, 1]
    """
    template = get_template(template_name)
    weights = template["weights"]

    reward = 0.0
    for component, weight in weights.items():
        score = scores.get(component, 0.0)
        reward += weight * score

    return max(0.0, min(1.0, reward))


# ─── Convenience functions for each template ─────────────────────────────────


def temporal_atomic_v1(ans: float, temporal: float, fmt: float) -> float:
    return compute_reward("temporal_atomic_v1", {"ans": ans, "temporal": temporal, "format": fmt})


def count_order_v1(ans: float, order_or_count: float, support: float, fmt: float) -> float:
    return compute_reward("count_order_v1", {
        "ans": ans, "order_or_count": order_or_count, "support": support, "format": fmt
    })


def grounding_v1(ans: float, span_iou: float, contrast: float, fmt: float) -> float:
    return compute_reward("grounding_v1", {
        "ans": ans, "span_iou": span_iou, "contrast": contrast, "format": fmt
    })


def long_retrieval_v1(ans: float, retrieval: float, support: float, efficiency: float) -> float:
    return compute_reward("long_retrieval_v1", {
        "ans": ans, "retrieval": retrieval, "support": support, "efficiency": efficiency
    })


def causal_relation_v1(ans: float, logic: float, support: float, fmt: float) -> float:
    return compute_reward("causal_relation_v1", {
        "ans": ans, "logic": logic, "support": support, "format": fmt
    })


def future_pred_v1(ans: float, future_slot: float, contrast: float, fmt: float) -> float:
    return compute_reward("future_pred_v1", {
        "ans": ans, "future_slot": future_slot, "contrast": contrast, "format": fmt
    })


def ego_intent_v1(what: float, why: float, next_step: float, fmt: float) -> float:
    return compute_reward("ego_intent_v1", {
        "what": what, "why": why, "next": next_step, "format": fmt
    })


def spatial_video_v1(ans: float, spatial: float, support: float, fmt: float) -> float:
    return compute_reward("spatial_video_v1", {
        "ans": ans, "spatial": spatial, "support": support, "format": fmt
    })


def av_fusion_v1(ans: float, cross_modal: float, ablation_consistency: float, fmt: float) -> float:
    return compute_reward("av_fusion_v1", {
        "ans": ans, "cross_modal": cross_modal,
        "ablation_consistency": ablation_consistency, "format": fmt
    })


def knowledge_acq_v1(ans: float, support: float, adaptation: float, fmt: float) -> float:
    return compute_reward("knowledge_acq_v1", {
        "ans": ans, "support": support, "adaptation": adaptation, "format": fmt
    })


def anti_shortcut_v1(clean_ans: float, perturbed_ans: float, contrast_gap: float) -> float:
    return compute_reward("anti_shortcut_v1", {
        "clean_ans": clean_ans, "perturbed_ans": perturbed_ans, "contrast_gap": contrast_gap
    })
