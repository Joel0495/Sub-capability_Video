"""Source adapter for NExT-QA (train split).

HF repo: lmms-lab/NExTQA
Multi-event causal video QA, ~35k train samples.
"""

import json
import csv
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


class NExTQAAdapter(BaseAdapter):
    dataset_name = "nextqa"
    display_name = "NExT-QA"
    hf_repo = "lmms-lab/NExTQA"
    license = "BSD"
    is_rl_native = False

    # NExT-QA question type to sub-ability mapping
    QTYPE_MAP = {
        "CW": "causal_why_how",     # Causal Why
        "CH": "causal_why_how",     # Causal How
        "TN": "temporal_relation",  # Temporal Next
        "TC": "temporal_relation",  # Temporal Current
        "TP": "temporal_relation",  # Temporal Previous
        "DC": "interaction_logic",  # Descriptive Count
        "DL": "interaction_logic",  # Descriptive Location
        "DO": "interaction_logic",  # Descriptive Other
    }

    def _post_download_commands(self) -> str:
        return (
            f"# NExT-QA: move annotation CSVs to {self.ann_dir}/\n"
            f"# Videos should be in {self.video_dir}/\n"
            "# Expected files: train.csv, val.csv (test.csv is forbidden)"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        # Try CSV format first (common for NExT-QA)
        csv_path = self.ann_dir / f"{split}.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield dict(row)
            return

        # Try JSON format
        json_path = self.ann_dir / f"{split}.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            return

        # Try JSONL
        jsonl_path = self.ann_dir / f"{split}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.CAUSAL_RELATION_REASONING,
        task_type: TaskType = TaskType.MCQ,
    ) -> CanonicalSample:
        video_id = raw.get("video", raw.get("video_id", ""))
        question = raw.get("question", "")
        answer = str(raw.get("answer", ""))
        raw_id = raw.get("qid", raw.get("id", video_id))
        qtype = raw.get("type", raw.get("qtype", ""))

        # Build options from a0-a4 fields (NExT-QA format)
        options = []
        for i in range(5):
            opt = raw.get(f"a{i}", "")
            if opt:
                options.append(opt)

        if options:
            opt_labels = "ABCDE"
            opt_text = " ".join(
                f"{opt_labels[i]}. {opt}" for i, opt in enumerate(options)
            )
            question_text = f"Question: {question} Options: {opt_text}. Answer with one capital letter."
            # Convert numeric answer to letter
            try:
                answer_idx = int(answer)
                answer = opt_labels[answer_idx]
            except (ValueError, IndexError):
                pass
        else:
            question_text = f"Question: {question}"

        video_url = self.video_path(str(video_id))

        messages = self.make_messages(
            video_url=video_url,
            question_text=question_text,
            answer_text=answer,
        )

        graders = [self.make_mcq_grader(answer)]

        sub_ability = self.QTYPE_MAP.get(qtype, "causal_why_how")

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability="Causal",
                datasource=self.display_name,
                sub_ability=[sub_ability],
                task_type=task_type,
                build_info=BuildInfo(
                    build_type=BuildType.CONVERTED,
                    raw_ann_path=str(self.ann_dir / "train.csv"),
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template="causal_relation_v1",
                    weights={"ans": 0.6, "logic": 0.2, "support": 0.1, "format": 0.1},
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=sub_capability.value,
                ),
            ),
        )
