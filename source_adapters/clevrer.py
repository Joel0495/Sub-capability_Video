"""Source adapter for CLEVRER (train split).

HF repo: dali-does/CLEVRER
Synthetic video reasoning, ~20k train samples.
Serves: causal, future prediction, spatial, anti-shortcut.
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


class CLEVRERAdapter(BaseAdapter):
    dataset_name = "clevrer"
    display_name = "CLEVRER"
    hf_repo = "dali-does/CLEVRER"
    license = "MIT"
    is_rl_native = False

    # CLEVRER question types
    QTYPE_MAP = {
        "descriptive": SubCapability.CAUSAL_RELATION_REASONING,
        "explanatory": SubCapability.CAUSAL_RELATION_REASONING,
        "predictive": SubCapability.FUTURE_EVENT_COUNTERFACTUAL,
        "counterfactual": SubCapability.FUTURE_EVENT_COUNTERFACTUAL,
    }

    def _post_download_commands(self) -> str:
        return (
            f"# CLEVRER: synthetic video reasoning\n"
            f"# Videos are rendered scenes in {self.video_dir}/\n"
            f"# Annotations in {self.ann_dir}/"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        # CLEVRER may have nested structure: each video has multiple questions
        for json_file in sorted(self.ann_dir.glob(f"*{split}*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for video_entry in data:
                        questions = video_entry.get("questions", [])
                        video_id = video_entry.get("scene_index",
                                    video_entry.get("video_filename", ""))
                        for q in questions:
                            q["video_id"] = video_id
                            yield q
            except (json.JSONDecodeError, IOError):
                continue

        # Try flat format
        for jsonl_file in sorted(self.ann_dir.glob(f"*{split}*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            except (json.JSONDecodeError, IOError):
                continue

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.CAUSAL_RELATION_REASONING,
        task_type: TaskType = None,
    ) -> CanonicalSample:
        video_id = str(raw.get("video_id", raw.get("scene_index", "")))
        question = raw.get("question", raw.get("question_body", ""))
        raw_id = raw.get("question_id", raw.get("id", video_id))

        # Detect question type
        qtype = raw.get("question_type", "descriptive").lower()
        detected_cap = self.QTYPE_MAP.get(qtype, sub_capability)

        # Handle choices
        choices = raw.get("choices", raw.get("options", []))
        if choices:
            if task_type is None:
                task_type = TaskType.MCQ
            opt_labels = "ABCDEFGH"
            if isinstance(choices[0], dict):
                choice_texts = [c.get("choice", c.get("text", str(c))) for c in choices]
                # Find correct answer
                answer_idx = next(
                    (i for i, c in enumerate(choices) if c.get("answer") == "correct"),
                    0
                )
            else:
                choice_texts = [str(c) for c in choices]
                answer_idx = int(raw.get("answer", 0))
            opt_text = " ".join(
                f"{opt_labels[i]}. {ct}" for i, ct in enumerate(choice_texts)
            )
            question_text = f"Question: {question} Options: {opt_text}. Answer with one capital letter."
            answer = opt_labels[answer_idx] if answer_idx < len(opt_labels) else "A"
        else:
            if task_type is None:
                task_type = TaskType.OPEN_ENDED
            question_text = f"Question: {question}"
            answer = str(raw.get("answer", ""))

        video_url = self.video_path(video_id)
        messages = self.make_messages(
            video_url=video_url,
            question_text=question_text,
            answer_text=answer,
        )

        graders = [self.make_mcq_grader(answer) if task_type == TaskType.MCQ
                    else self.make_exact_match_grader(answer)]

        reward_map = {
            SubCapability.CAUSAL_RELATION_REASONING: "causal_relation_v1",
            SubCapability.FUTURE_EVENT_COUNTERFACTUAL: "future_pred_v1",
            SubCapability.SPATIAL_SPATIOTEMPORAL_REASONING: "spatial_video_v1",
        }

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability=detected_cap.value.split("_")[0].capitalize(),
                datasource=self.display_name,
                sub_ability=[qtype],
                task_type=task_type,
                build_info=BuildInfo(
                    build_type=BuildType.CONVERTED,
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template=reward_map.get(detected_cap, "causal_relation_v1"),
                    weights={"ans": 0.6, "logic": 0.2, "support": 0.1, "format": 0.1},
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=detected_cap.value,
                ),
            ),
        )
