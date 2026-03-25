"""Tests for the canonical schema."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError
from schema.canonical import (
    CanonicalSample,
    VideoURL,
    TextContent,
    VideoURLContent,
    Message,
    Grader,
    GraderParams,
    DataInfo,
    BuildInfo,
    BuildType,
    TaskType,
    ExtraInfo,
    RewardInfo,
    SamplingInfo,
)


def _make_valid_sample(**overrides) -> dict:
    """Create a valid sample dict for testing."""
    base = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a video reasoning assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///data/raw/test/videos/001.mp4"},
                    },
                    {"type": "text", "text": "Question: What happens first? A. sit B. stand"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "A"}],
            },
        ],
        "graders": [
            {"type": "ruler", "name": "option_match", "gt": "A", "params": {"min_score": 1.0}},
        ],
        "data_info": {
            "data_id": "test_001",
            "ability": "Temporal",
            "datasource": "TestDataset",
            "task_type": "mcq",
            "sub_ability": ["event_order"],
            "build_info": {"build_type": "converted"},
        },
    }
    base.update(overrides)
    return base


class TestCanonicalSample:
    def test_valid_sample(self):
        data = _make_valid_sample()
        sample = CanonicalSample.model_validate(data)
        assert sample.data_info.data_id == "test_001"
        assert sample.data_info.task_type == TaskType.MCQ
        assert len(sample.graders) == 1
        assert len(sample.messages) == 3

    def test_missing_user_message_rejected(self):
        data = _make_valid_sample()
        data["messages"] = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
        ]
        with pytest.raises(ValidationError, match="user"):
            CanonicalSample.model_validate(data)

    def test_missing_assistant_message_rejected(self):
        data = _make_valid_sample()
        data["messages"] = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "Q?"}]},
        ]
        with pytest.raises(ValidationError, match="assistant"):
            CanonicalSample.model_validate(data)

    def test_empty_graders_rejected(self):
        data = _make_valid_sample()
        data["graders"] = []
        with pytest.raises(ValidationError, match="grader"):
            CanonicalSample.model_validate(data)

    def test_with_extra_info(self):
        data = _make_valid_sample()
        data["extra_info"] = {
            "pass_rate": 0.42,
            "reward_info": {
                "reward_template": "temporal_atomic_v1",
                "weights": {"ans": 0.7, "temporal": 0.2, "format": 0.1},
            },
            "sampling_info": {
                "mix_bucket": "temporal_atomic",
                "curriculum_stage": "warmup",
            },
        }
        sample = CanonicalSample.model_validate(data)
        assert sample.extra_info.pass_rate == 0.42
        assert sample.extra_info.reward_info.reward_template == "temporal_atomic_v1"


class TestVideoURL:
    def test_valid_file_url(self):
        url = VideoURL(url="file:///data/raw/test/video.mp4")
        assert url.url.startswith("file://")

    def test_base64_rejected(self):
        with pytest.raises(ValidationError, match="base64"):
            VideoURL(url="data:video/mp4;base64,AAAA...")

    def test_with_clip_params(self):
        url = VideoURL(
            url="file:///data/raw/test/video.mp4",
            clip_fps=1.0,
            start_s=5.0,
            end_s=10.0,
        )
        assert url.clip_fps == 1.0
        assert url.start_s == 5.0


class TestDataInfo:
    def test_build_type_values(self):
        for bt in ["native", "converted", "synthetic", "paired"]:
            info = DataInfo(
                data_id="test",
                ability="Temporal",
                datasource="Test",
                task_type="mcq",
                build_info=BuildInfo(build_type=bt),
            )
            assert info.build_info.build_type.value == bt

    def test_invalid_build_type(self):
        with pytest.raises(ValidationError):
            DataInfo(
                data_id="test",
                ability="Temporal",
                datasource="Test",
                task_type="mcq",
                build_info=BuildInfo(build_type="invalid"),
            )
