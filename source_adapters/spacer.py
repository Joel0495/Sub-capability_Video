"""Source adapter for SpaceR-151k (RL-native).

HF repo: RUBBISHLIKE/SpaceR-151k
91k spatial reasoning (ScanNet) + 60k general understanding (from Video-R1).
"""

import json
from pathlib import Path
from typing import Iterator

from source_adapters.base_adapter import BaseAdapter
from schema.canonical import (
    CanonicalSample,
    SubCapability,
    TaskType,
    BuildType,
    DataInfo,
    BuildInfo,
    ExtraInfo,
    RewardInfo,
    SamplingInfo,
)


class SpaceRAdapter(BaseAdapter):
    dataset_name = "spacer"
    display_name = "SpaceR-151k"
    hf_repo = "RUBBISHLIKE/SpaceR-151k"
    license = "MIT"
    is_rl_native = True

    def _post_download_commands(self) -> str:
        return (
            "# SpaceR stores annotations as JSON\n"
            "# SR-91k: spatial reasoning from ScanNet indoor scenes\n"
            "# 60k: general understanding from Video-R1\n"
            f"# Organize into {self.raw_dir}/annotations/"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        for json_file in sorted(self.raw_dir.glob("**/*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
            except (json.JSONDecodeError, IOError):
                continue

    def _detect_task_type(self, raw: dict) -> TaskType:
        task = raw.get("task_type", "").lower()
        if "multi" in task or "choice" in task:
            return TaskType.MCQ
        if "regression" in task:
            return TaskType.REGRESSION
        if "count" in task:
            return TaskType.COUNT
        if raw.get("options") or raw.get("choices"):
            return TaskType.MCQ
        return TaskType.OPEN_ENDED

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.SPATIAL_SPATIOTEMPORAL_REASONING,
        task_type: TaskType = None,
    ) -> CanonicalSample:
        if task_type is None:
            task_type = self._detect_task_type(raw)

        video_id = raw.get("video_path", raw.get("video", raw.get("video_id", "")))
        question = raw.get("question", raw.get("prompt", ""))
        answer = str(raw.get("answer", raw.get("response", "")))
        raw_id = raw.get("id", video_id)

        options = raw.get("options", raw.get("choices", []))
        if options and task_type == TaskType.MCQ:
            opt_labels = "ABCDEFGH"
            opt_text = " ".join(
                f"{opt_labels[i]}. {opt}" for i, opt in enumerate(options)
            )
            question = f"{question} Options: {opt_text}. Answer with one capital letter."

        if video_id and not video_id.startswith("file://"):
            video_url = self.video_path(video_id.replace(".mp4", ""), "mp4")
        else:
            video_url = video_id or "file:///placeholder"

        messages = self.make_messages(
            video_url=video_url,
            question_text=question,
            answer_text=answer,
        )

        graders = [self.make_mcq_grader(answer) if task_type == TaskType.MCQ
                    else self.make_exact_match_grader(answer)]

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability="Spatial",
                datasource=self.display_name,
                sub_ability=["relative_position", "motion_in_space"],
                task_type=task_type,
                build_info=BuildInfo(
                    build_type=BuildType.NATIVE,
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template="spatial_video_v1",
                    weights={"ans": 0.5, "spatial": 0.25, "support": 0.15, "format": 0.10},
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=sub_capability.value,
                ),
            ),
        )
