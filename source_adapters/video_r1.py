"""Source adapter for Video-R1-260k (RL-native).

HF repo: Video-R1/Video-R1-data
Contains samples from CLEVRER, NExT-QA, STAR, PerceptionTest, LLaVA-Video.
260k samples (44% video, 56% image).
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


class VideoR1Adapter(BaseAdapter):
    dataset_name = "video_r1"
    display_name = "Video-R1-260k"
    hf_repo = "Video-R1/Video-R1-data"
    license = "Apache-2.0"
    is_rl_native = True

    # Sub-dataset to sub-capability mapping
    SOURCE_CAPABILITY_MAP = {
        "clevrer": SubCapability.CAUSAL_RELATION_REASONING,
        "nextqa": SubCapability.CAUSAL_RELATION_REASONING,
        "star": SubCapability.CAUSAL_RELATION_REASONING,
        "perception_test": SubCapability.TEMPORAL_ATOMIC,
        "llava_video": SubCapability.TEMPORAL_ATOMIC,
    }

    def _post_download_commands(self) -> str:
        return (
            "# Video-R1 stores annotations as JSON files\n"
            "# Videos are referenced from sub-dataset directories\n"
            f"# Annotations will be in {self.raw_dir}/\n"
            f"# Organize video links as needed"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        """Iterate over Video-R1 annotation files.

        Video-R1 stores data as JSON with fields like:
        - video_path, question, answer, source, task_type
        """
        ann_dir = self.raw_dir
        # Video-R1 may have multiple JSON files (train, cot, etc.)
        for json_file in sorted(ann_dir.glob("**/*.json")):
            if "cot" in json_file.name.lower():
                continue  # Skip CoT file (separate adapter or later)
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
                elif isinstance(data, dict):
                    for key, item in data.items():
                        if isinstance(item, dict):
                            item["_key"] = key
                            yield item
            except (json.JSONDecodeError, IOError):
                continue

    def _detect_sub_capability(self, raw: dict) -> SubCapability:
        """Detect which sub-capability a sample belongs to."""
        source = raw.get("source", "").lower()
        for key, cap in self.SOURCE_CAPABILITY_MAP.items():
            if key in source:
                return cap
        return SubCapability.TEMPORAL_ATOMIC  # default

    def _detect_task_type(self, raw: dict) -> TaskType:
        """Detect task type from raw annotation."""
        task = raw.get("task_type", "").lower()
        if "multi" in task or "choice" in task or "mcq" in task:
            return TaskType.MCQ
        if "count" in task or "number" in task:
            return TaskType.COUNT
        if "regression" in task:
            return TaskType.REGRESSION
        if raw.get("options") or raw.get("choices"):
            return TaskType.MCQ
        return TaskType.OPEN_ENDED

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = None,
        task_type: TaskType = None,
    ) -> CanonicalSample:
        if sub_capability is None:
            sub_capability = self._detect_sub_capability(raw)
        if task_type is None:
            task_type = self._detect_task_type(raw)

        video_id = raw.get("video_path", raw.get("video", raw.get("video_id", "")))
        question = raw.get("question", raw.get("prompt", ""))
        answer = str(raw.get("answer", raw.get("response", "")))
        raw_id = raw.get("id", raw.get("_key", video_id))

        # Format question with options if MCQ
        options = raw.get("options", raw.get("choices", []))
        if options and task_type == TaskType.MCQ:
            opt_labels = "ABCDEFGH"
            opt_text = " ".join(
                f"{opt_labels[i]}. {opt}" for i, opt in enumerate(options)
            )
            question = f"{question} Options: {opt_text}. Answer with one capital letter."

        # Build video URL
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

        # Map sub_capability to reward template
        reward_map = {
            SubCapability.TEMPORAL_ATOMIC: "temporal_atomic_v1",
            SubCapability.TEMPORAL_COUNT_ORDER: "count_order_v1",
            SubCapability.CAUSAL_RELATION_REASONING: "causal_relation_v1",
            SubCapability.ANTI_SHORTCUT_CONTRAST: "anti_shortcut_v1",
        }
        reward_template = reward_map.get(sub_capability, "temporal_atomic_v1")

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability=sub_capability.value.split("_")[0].capitalize(),
                datasource=self.display_name,
                sub_ability=[sub_capability.value],
                task_type=task_type,
                build_info=BuildInfo(
                    build_type=BuildType.NATIVE,
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template=reward_template,
                    weights={"ans": 0.7, "temporal": 0.2, "format": 0.1},
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=sub_capability.value,
                ),
            ),
        )
