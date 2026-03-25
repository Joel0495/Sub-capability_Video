"""Source adapter for STAR (train split).

HF repo: STAR-Benchmark/STAR
Situated reasoning in real-world videos, ~45k train samples.
25% sequencing questions with temporal focus.
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


class STARAdapter(BaseAdapter):
    dataset_name = "star"
    display_name = "STAR"
    hf_repo = "STAR-Benchmark/STAR"
    license = "MIT"
    is_rl_native = False

    # STAR question types
    QTYPE_CAPABILITY = {
        "Interaction": SubCapability.CAUSAL_RELATION_REASONING,
        "Sequence": SubCapability.CAUSAL_RELATION_REASONING,
        "Prediction": SubCapability.FUTURE_EVENT_COUNTERFACTUAL,
        "Feasibility": SubCapability.CAUSAL_RELATION_REASONING,
    }

    def _post_download_commands(self) -> str:
        return (
            f"# STAR: situated reasoning annotations\n"
            f"# Move JSON annotations to {self.ann_dir}/\n"
            f"# Videos from Charades to {self.video_dir}/"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        # STAR stores data per question type
        for json_file in sorted(self.ann_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else data.get(split, data.get("data", []))
                if isinstance(items, list):
                    for item in items:
                        yield item
            except (json.JSONDecodeError, IOError):
                continue

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.CAUSAL_RELATION_REASONING,
        task_type: TaskType = TaskType.MCQ,
    ) -> CanonicalSample:
        video_id = raw.get("video_id", raw.get("video", ""))
        question = raw.get("question", "")
        answer = str(raw.get("answer", ""))
        raw_id = raw.get("question_id", raw.get("id", video_id))
        qtype = raw.get("question_type", raw.get("type", ""))

        # Build options
        choices = raw.get("choices", [])
        if choices:
            opt_labels = "ABCDEFGH"
            if isinstance(choices[0], dict):
                choice_texts = [c.get("choice", c.get("text", str(c))) for c in choices]
            else:
                choice_texts = [str(c) for c in choices]
            opt_text = " ".join(
                f"{opt_labels[i]}. {ct}" for i, ct in enumerate(choice_texts)
            )
            question_text = f"Question: {question} Options: {opt_text}. Answer with one capital letter."
            # Convert answer to letter if numeric
            try:
                answer_idx = int(answer)
                answer = opt_labels[answer_idx]
            except (ValueError, IndexError):
                pass
        else:
            question_text = f"Question: {question}"

        # Detect capability from question type
        detected_cap = self.QTYPE_CAPABILITY.get(qtype, sub_capability)
        video_url = self.video_path(str(video_id))

        messages = self.make_messages(
            video_url=video_url,
            question_text=question_text,
            answer_text=answer,
        )

        graders = [self.make_mcq_grader(answer)]

        reward_map = {
            SubCapability.CAUSAL_RELATION_REASONING: ("causal_relation_v1",
                {"ans": 0.6, "logic": 0.2, "support": 0.1, "format": 0.1}),
            SubCapability.FUTURE_EVENT_COUNTERFACTUAL: ("future_pred_v1",
                {"ans": 0.5, "future_slot": 0.2, "contrast": 0.2, "format": 0.1}),
        }
        template, weights = reward_map.get(
            detected_cap,
            ("causal_relation_v1", {"ans": 0.6, "logic": 0.2, "support": 0.1, "format": 0.1})
        )

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability="Causal" if detected_cap == SubCapability.CAUSAL_RELATION_REASONING else "Prediction",
                datasource=self.display_name,
                sub_ability=[qtype.lower() if qtype else "situated_reasoning"],
                task_type=task_type,
                build_info=BuildInfo(
                    build_type=BuildType.CONVERTED,
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template=template,
                    weights=weights,
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=detected_cap.value,
                ),
            ),
        )
