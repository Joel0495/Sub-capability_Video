"""Source adapter for NEP V1-33K (RL-native).

HF repo: haonan3/V1-33K
33k (past + future) video pairs for next-event prediction.
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


class NEPV1Adapter(BaseAdapter):
    dataset_name = "nep_v1"
    display_name = "NEP V1-33K"
    hf_repo = "haonan3/V1-33K"
    license = "MIT"
    is_rl_native = True

    def _post_download_commands(self) -> str:
        return (
            "# NEP V1-33K: next-event prediction pairs\n"
            "# Contains past/future video segment pairs\n"
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

        # Also try JSONL format
        for jsonl_file in sorted(self.raw_dir.glob("**/*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            except (json.JSONDecodeError, IOError):
                continue

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.FUTURE_EVENT_COUNTERFACTUAL,
        task_type: TaskType = None,
    ) -> CanonicalSample:
        if task_type is None:
            task_type = TaskType.MCQ if raw.get("options") else TaskType.OPEN_ENDED

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
                ability="Prediction",
                datasource=self.display_name,
                sub_ability=["what_happens_next", "future_event_summary"],
                task_type=task_type,
                build_info=BuildInfo(
                    build_type=BuildType.NATIVE,
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template="future_pred_v1",
                    weights={"ans": 0.5, "future_slot": 0.2, "contrast": 0.2, "format": 0.1},
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=sub_capability.value,
                ),
            ),
        )
