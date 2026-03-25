"""Tests for benchmark hold-out guard."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from registry import (
    is_benchmark,
    assert_not_benchmark,
    assert_valid_split,
    get_allowed_splits,
    load_benchmark_holdout,
)


class TestIsBenchmark:
    def test_all_benchmarks_detected(self):
        benchmarks = [
            "TempCompass",
            "TemporalBench",
            "MVBench",
            "MLVU",
            "LongVideoBench",
            "Video-MME",
            "EgoSchema",
            "PerceptionTest",
            "Video-MMMU",
            "VSI-Bench",
            "EgoIntent",
            "TutorialVQA",
        ]
        for name in benchmarks:
            assert is_benchmark(name), f"{name} should be detected as benchmark"

    def test_normalized_names(self):
        # "TempCompass" -> "tempcompass", "Video-MME" -> "video_mme"
        assert is_benchmark("tempcompass")
        assert is_benchmark("video_mme")
        assert is_benchmark("vsi_bench")
        assert is_benchmark("perceptiontest")

    def test_training_datasets_not_blocked(self):
        allowed = [
            "nextqa",
            "star",
            "clevrer",
            "tvqa",
            "qvhighlights",
            "ego4d",
            "video_r1",
            "spacer",
            "nep_v1",
            "activitynet_captions",
        ]
        for name in allowed:
            assert not is_benchmark(name), f"{name} should NOT be blocked"


class TestAssertNotBenchmark:
    def test_raises_for_benchmark(self):
        with pytest.raises(ValueError, match="BLOCKED"):
            assert_not_benchmark("TempCompass")

    def test_passes_for_training(self):
        assert_not_benchmark("nextqa")  # should not raise


class TestSplitRestrictions:
    def test_train_split_allowed(self):
        assert_valid_split("NExT-QA", "train")  # should not raise
        assert_valid_split("STAR", "train")

    def test_test_split_forbidden(self):
        with pytest.raises(ValueError, match="forbidden"):
            assert_valid_split("NExT-QA", "test")

    def test_val_split_allowed(self):
        assert_valid_split("NExT-QA", "val")

    def test_unrestricted_dataset(self):
        # RL-native datasets have no split restrictions
        assert_valid_split("video_r1", "train")  # should not raise
        assert_valid_split("video_r1", "test")   # no restriction on RL-native

    def test_get_allowed_splits(self):
        splits = get_allowed_splits("NExT-QA")
        assert splits is not None
        assert "train" in splits
        assert "val" in splits

    def test_get_allowed_splits_unrestricted(self):
        splits = get_allowed_splits("video_r1")
        assert splits is None  # no restrictions
