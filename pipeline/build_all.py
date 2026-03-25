"""Orchestrator: import all 11 capability builders, run each, and print stats."""

import importlib
import sys
from pathlib import Path

import fire

# The 11 sub-capability builder module names (one per capability).
# Each module must expose a class that subclasses BaseBuilder.
BUILDER_MODULES = [
    "builders.a_temporal_atomic",
    "builders.b_temporal_count_order",
    "builders.c_temporal_grounding",
    "builders.d_long_video_retrieval_memory",
    "builders.e_causal_relation_reasoning",
    "builders.f_future_event_counterfactual",
    "builders.g_egocentric_intent_next_step",
    "builders.h_spatial_spatiotemporal_reasoning",
    "builders.i_multimodal_av_fusion",
    "builders.j_topic_plot_knowledge_acquisition",
    "builders.k_anti_shortcut_contrast",
]


def _discover_builder_class(module):
    """Find the first BaseBuilder subclass in a module."""
    from builders.base import BaseBuilder

    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseBuilder)
            and obj is not BaseBuilder
        ):
            return obj
    return None


def build_all(data_root: str = "data", split: str = "train"):
    """Run all 11 capability builders and print aggregate stats.

    Args:
        data_root: Root data directory.
        split: Dataset split to build from.
    """
    all_stats = []
    total_samples = 0
    errors = []

    for mod_name in BUILDER_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            print(f"[SKIP] Module not found: {mod_name}")
            errors.append(mod_name)
            continue

        builder_cls = _discover_builder_class(mod)
        if builder_cls is None:
            print(f"[SKIP] No BaseBuilder subclass in {mod_name}")
            errors.append(mod_name)
            continue

        print(f"[BUILD] {mod_name} -> {builder_cls.__name__}")
        try:
            builder = builder_cls(data_root=data_root)
            stats = builder.build(split=split)
            all_stats.append(stats)
            total_samples += stats["total"]
            print(f"  -> {stats['total']} samples from {len(stats['sources'])} sources")
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors.append(mod_name)

    # Print summary
    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    for s in all_stats:
        print(f"  {s['capability']:45s} {s['total']:>8d} samples")
    print("-" * 60)
    print(f"  {'TOTAL':45s} {total_samples:>8d} samples")
    if errors:
        print(f"\n  Errors/skipped: {len(errors)}")
        for e in errors:
            print(f"    - {e}")

    return all_stats


if __name__ == "__main__":
    fire.Fire(build_all)
