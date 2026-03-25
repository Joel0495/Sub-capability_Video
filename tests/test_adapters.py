"""Tests for source adapters."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from source_adapters.base_adapter import BaseAdapter
from schema.canonical import SubCapability, TaskType, CanonicalSample


class ConcreteAdapter(BaseAdapter):
    """Concrete adapter for testing BaseAdapter methods."""
    dataset_name = "test_dataset"
    display_name = "Test Dataset"
    hf_repo = "test/test-dataset"
    license = "MIT"

    def iterate_raw(self, split="train"):
        yield {
            "video_id": "video_001",
            "question": "What happens?",
            "answer": "A",
            "id": "q001",
        }

    def to_canonical(self, raw, sub_capability=None, task_type=None):
        video_url = self.video_path(raw["video_id"])
        messages = self.make_messages(
            video_url=video_url,
            question_text=raw["question"],
            answer_text=raw["answer"],
        )
        from schema.canonical import DataInfo, BuildInfo, BuildType
        return CanonicalSample(
            messages=messages,
            graders=[self.make_mcq_grader(raw["answer"])],
            data_info=DataInfo(
                data_id=self.make_data_id(raw["id"]),
                ability="Test",
                datasource=self.display_name,
                task_type=TaskType.MCQ,
                build_info=BuildInfo(build_type=BuildType.CONVERTED),
            ),
        )


class TestBaseAdapter:
    def setup_method(self):
        self.adapter = ConcreteAdapter(data_root="/tmp/test_data")

    def test_video_path(self):
        path = self.adapter.video_path("video_001")
        assert path.startswith("file://")
        assert "video_001.mp4" in path
        assert "/tmp/test_data/raw/test_dataset/videos/" in path

    def test_video_path_custom_ext(self):
        path = self.adapter.video_path("video_001", ext="avi")
        assert "video_001.avi" in path

    def test_make_data_id(self):
        data_id = self.adapter.make_data_id("q001")
        assert data_id == "test_dataset_q001"

    def test_make_messages(self):
        messages = self.adapter.make_messages(
            video_url="file:///test/video.mp4",
            question_text="What happens?",
            answer_text="A",
        )
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    def test_make_messages_with_subtitle(self):
        messages = self.adapter.make_messages(
            video_url="file:///test/video.mp4",
            question_text="What happens?",
            answer_text="A",
            subtitle_text="Some subtitle context",
        )
        # User message should have 3 content items: video, question, subtitle
        user_msg = messages[1]
        assert len(user_msg.content) == 3

    def test_make_mcq_grader(self):
        grader = self.adapter.make_mcq_grader("B")
        assert grader.name == "option_match"
        assert grader.gt == "B"
        assert grader.type == "ruler"

    def test_make_exact_match_grader(self):
        grader = self.adapter.make_exact_match_grader("42")
        assert grader.name == "exact_match"
        assert grader.gt == "42"

    def test_to_canonical_produces_valid_sample(self):
        for raw in self.adapter.iterate_raw():
            sample = self.adapter.to_canonical(raw)
            assert isinstance(sample, CanonicalSample)
            assert sample.data_info.data_id == "test_dataset_q001"
            assert sample.data_info.datasource == "Test Dataset"

    def test_generate_download_script(self):
        script = self.adapter.generate_download_script()
        assert "hf_dl.sh" in script or "hf_dl" in script
        assert "test/test-dataset" in script

    def test_build_type(self):
        assert self.adapter.build_type().value == "converted"

    def test_rl_native_build_type(self):
        self.adapter.is_rl_native = True
        assert self.adapter.build_type().value == "native"
        self.adapter.is_rl_native = False  # reset


class TestManualDownloadAdapter:
    def test_manual_download_generates_instructions(self):
        adapter = ConcreteAdapter(data_root="/tmp/test_data")
        adapter.hf_repo = None  # Force manual download
        script = adapter.generate_download_script()
        assert "MANUAL DOWNLOAD" in script
        assert "mkdir -p" in script
