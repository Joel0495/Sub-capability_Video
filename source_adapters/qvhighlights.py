"""Source adapter for QVHighlights (train split).

HF repo: jayleicn/QVHighlights
Moment retrieval + highlight detection, ~10k train samples.
Each sample has natural language queries with temporal moment annotations.
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
    Evidence,
    ExtraInfo,
    RewardInfo,
    SamplingInfo,
    Grader,
    GraderParams,
)


class QVHighlightsAdapter(BaseAdapter):
    dataset_name = "qvhighlights"
    display_name = "QVHighlights"
    hf_repo = "jayleicn/QVHighlights"
    license = "MIT"
    is_rl_native = False

    def _post_download_commands(self) -> str:
        return (
            f"# QVHighlights: moment retrieval dataset\n"
            f"# Move highlight_train_release.jsonl to {self.ann_dir}/\n"
            f"# Videos to {self.video_dir}/"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        # QVHighlights uses JSONL format
        for jsonl_file in sorted(self.ann_dir.glob(f"*{split}*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            except (json.JSONDecodeError, IOError):
                continue

        # Also try JSON
        for json_file in sorted(self.ann_dir.glob(f"*{split}*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
            except (json.JSONDecodeError, IOError):
                continue

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.TEMPORAL_GROUNDING,
        task_type: TaskType = TaskType.GROUNDING,
    ) -> CanonicalSample:
        video_id = raw.get("vid", raw.get("video_id", ""))
        query = raw.get("query", raw.get("question", ""))
        raw_id = raw.get("qid", raw.get("id", video_id))
        duration = raw.get("duration", None)

        # Relevant windows (temporal spans)
        relevant_windows = raw.get("relevant_windows", [])
        if relevant_windows:
            # Format answer as span
            spans_text = "; ".join(
                f"[{s[0]:.1f}, {s[1]:.1f}]" for s in relevant_windows
            )
            answer = spans_text
        else:
            answer = "N/A"

        question_text = (
            f"Question: Locate the moment in the video that matches this description: "
            f"\"{query}\". Answer with the start and end timestamps in seconds."
        )

        video_url = self.video_path(str(video_id))
        messages = self.make_messages(
            video_url=video_url,
            question_text=question_text,
            answer_text=answer,
        )

        # Grounding uses span_iou grader
        graders = [
            Grader(
                type="ruler",
                name="span_iou",
                gt=answer,
                params=GraderParams(min_score=0.5),
            )
        ]

        # Evidence from relevant windows
        evidence = Evidence(
            support_spans=relevant_windows if relevant_windows else None,
        )

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability="Temporal",
                datasource=self.display_name,
                sub_ability=["moment_retrieval", "highlight_detection"],
                task_type=task_type,
                evidence=evidence,
                build_info=BuildInfo(
                    build_type=BuildType.CONVERTED,
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template="grounding_v1",
                    weights={"ans": 0.45, "span_iou": 0.40, "contrast": 0.10, "format": 0.05},
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=sub_capability.value,
                ),
            ),
        )
