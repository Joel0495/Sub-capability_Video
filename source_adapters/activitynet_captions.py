"""Source adapter for ActivityNet Captions (train split).

HF repo: HuggingFaceM4/ActivitiyNet_Captions
Activity captioning with temporal annotations, ~37k train segments.
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
    VideoProfile,
    ExtraInfo,
    RewardInfo,
    SamplingInfo,
    Grader,
    GraderParams,
)


class ActivityNetCaptionsAdapter(BaseAdapter):
    dataset_name = "activitynet_captions"
    display_name = "ActivityNet Captions"
    hf_repo = "HuggingFaceM4/ActivitiyNet_Captions"
    license = "MIT"
    is_rl_native = False

    def _post_download_commands(self) -> str:
        return (
            f"# ActivityNet Captions: dense captioning with temporal segments\n"
            f"# Move train.json to {self.ann_dir}/\n"
            f"# Videos to {self.video_dir}/"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        json_path = self.ann_dir / f"{split}.json"
        if not json_path.exists():
            # Try alternative names
            for name in ["train.json", "captions_train.json", "train_ids.json"]:
                alt = self.ann_dir / name
                if alt.exists():
                    json_path = alt
                    break

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)

            # ActivityNet format: {video_id: {timestamps: [...], sentences: [...]}}
            if isinstance(data, dict):
                for video_id, info in data.items():
                    if isinstance(info, dict):
                        timestamps = info.get("timestamps", [])
                        sentences = info.get("sentences", [])
                        duration = info.get("duration", None)
                        for i, (ts, sent) in enumerate(zip(timestamps, sentences)):
                            yield {
                                "video_id": video_id,
                                "timestamp": ts,
                                "sentence": sent,
                                "duration": duration,
                                "segment_idx": i,
                            }

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.TEMPORAL_GROUNDING,
        task_type: TaskType = TaskType.GROUNDING,
    ) -> CanonicalSample:
        video_id = raw.get("video_id", "")
        sentence = raw.get("sentence", "")
        timestamp = raw.get("timestamp", [0, 0])
        duration = raw.get("duration", None)
        seg_idx = raw.get("segment_idx", 0)
        raw_id = f"{video_id}_seg{seg_idx}"

        # Format as temporal grounding task
        start_s, end_s = timestamp[0], timestamp[1]
        answer = f"[{start_s:.1f}, {end_s:.1f}]"

        question_text = (
            f"Question: Locate the moment in the video described by: "
            f"\"{sentence}\". Answer with the start and end timestamps in seconds."
        )

        video_url = self.video_path(str(video_id))
        messages = self.make_messages(
            video_url=video_url,
            question_text=question_text,
            answer_text=answer,
            start_s=start_s,
            end_s=end_s,
        )

        graders = [
            Grader(
                type="ruler",
                name="span_iou",
                gt=answer,
                params=GraderParams(min_score=0.5),
            )
        ]

        evidence = Evidence(
            support_spans=[timestamp],
        )

        video_profile = None
        if duration:
            video_profile = VideoProfile(
                video_id=video_id,
                duration_s=duration,
                is_long_video=duration > 120,
            )

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(raw_id),
                ability="Temporal",
                datasource=self.display_name,
                sub_ability=["moment_retrieval", "support_span_localization"],
                task_type=task_type,
                video_profile=video_profile,
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
