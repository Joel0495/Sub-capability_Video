"""Source adapter for TVQA (train split).

HF repo: pengxiang/tvqa
TV show QA, 152.5K QA pairs from 21.8K video clips.
Shows: Friends, Big Bang Theory, How I Met Your Mother, etc.
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
    ModalityProfile,
    ExtraInfo,
    RewardInfo,
    SamplingInfo,
)


class TVQAAdapter(BaseAdapter):
    dataset_name = "tvqa"
    display_name = "TVQA"
    hf_repo = "pengxiang/tvqa"
    license = "MIT"
    is_rl_native = False

    def _post_download_commands(self) -> str:
        return (
            f"# TVQA: TV show video QA\n"
            f"# Move tvqa_train.jsonl to {self.ann_dir}/\n"
            f"# Video clips to {self.video_dir}/\n"
            f"# Subtitle files to {self.raw_dir}/subtitles/"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        for jsonl_file in sorted(self.ann_dir.glob(f"*{split}*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            except (json.JSONDecodeError, IOError):
                continue

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
        sub_capability: SubCapability = SubCapability.LONG_VIDEO_RETRIEVAL_MEMORY,
        task_type: TaskType = TaskType.MCQ,
    ) -> CanonicalSample:
        # TVQA fields
        vid_name = raw.get("vid_name", raw.get("video_id", ""))
        question = raw.get("q", raw.get("question", ""))
        raw_id = raw.get("qid", raw.get("id", vid_name))
        answer_idx = raw.get("answer_idx", 0)

        # TVQA has a0-a4 options
        options = []
        for i in range(5):
            opt = raw.get(f"a{i}", "")
            if opt:
                options.append(opt)

        subtitle = raw.get("located_sub_text", raw.get("subtitle", ""))
        ts = raw.get("ts", "")  # timestamp info

        opt_labels = "ABCDE"
        if options:
            opt_text = " ".join(
                f"{opt_labels[i]}. {opt}" for i, opt in enumerate(options)
            )
            question_text = f"Question: {question} Options: {opt_text}. Answer with one capital letter."
            try:
                answer = opt_labels[int(answer_idx)]
            except (ValueError, IndexError):
                answer = str(answer_idx)
        else:
            question_text = f"Question: {question}"
            answer = str(answer_idx)

        video_url = self.video_path(str(vid_name))
        messages = self.make_messages(
            video_url=video_url,
            question_text=question_text,
            answer_text=answer,
            subtitle_text=subtitle if subtitle else None,
        )

        graders = [self.make_mcq_grader(answer)]

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability="Memory",
                datasource=self.display_name,
                sub_ability=["subtitle_grounded_lookup", "long_context_memory"],
                task_type=task_type,
                modality_profile=ModalityProfile(
                    video=True,
                    subtitle=bool(subtitle),
                ),
                build_info=BuildInfo(
                    build_type=BuildType.CONVERTED,
                    video_path=video_url,
                ),
            ),
            extra_info=ExtraInfo(
                reward_info=RewardInfo(
                    reward_template="long_retrieval_v1",
                    weights={"ans": 0.35, "retrieval": 0.35, "support": 0.20, "efficiency": 0.10},
                ),
                sampling_info=SamplingInfo(
                    mix_bucket=sub_capability.value,
                ),
            ),
        )
