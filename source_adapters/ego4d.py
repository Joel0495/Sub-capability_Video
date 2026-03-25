"""Source adapter for Ego4D (train split).

Source: https://ego4d-data.org/ (manual download, requires license agreement)
3,670 hours egocentric video, multiple benchmark tasks.
Serves: long video memory, future prediction, egocentric intent, AV fusion.
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
    VideoProfile,
    ExtraInfo,
    RewardInfo,
    SamplingInfo,
)


class Ego4DAdapter(BaseAdapter):
    dataset_name = "ego4d"
    display_name = "Ego4D"
    hf_repo = None  # Manual download required
    license = "Ego4D License"
    is_rl_native = False

    def _generate_manual_download_instructions(self) -> str:
        return (
            f"# === Ego4D ({self.dataset_name}) ===\n"
            f"# MANUAL DOWNLOAD REQUIRED - License agreement needed\n"
            f"# 1. Visit https://ego4d-data.org/ and create account\n"
            f"# 2. Accept the Ego4D License Agreement\n"
            f"# 3. Install Ego4D CLI: pip install ego4d\n"
            f"# 4. Download videos:\n"
            f"#    python -m ego4d.cli.cli --output_directory {self.raw_dir} --datasets full_scale\n"
            f"# 5. Download annotations:\n"
            f"#    python -m ego4d.cli.cli --output_directory {self.raw_dir} --datasets annotations\n"
            f"# 6. Organize:\n"
            f"#    Videos -> {self.video_dir}/\n"
            f"#    Annotations -> {self.ann_dir}/\n"
            f"mkdir -p {self.video_dir} {self.ann_dir}\n"
        )

    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        # Ego4D has multiple annotation types (NLQ, MQ, FHO, etc.)
        for json_file in sorted(self.ann_dir.glob("**/*.json")):
            if "test" in json_file.name.lower():
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle different Ego4D annotation formats
                if isinstance(data, dict):
                    # NLQ/MQ format: {"videos": [...]}
                    videos = data.get("videos", data.get("clips", []))
                    if isinstance(videos, list):
                        for vid_entry in videos:
                            clips = vid_entry.get("clips", [vid_entry])
                            video_uid = vid_entry.get("video_uid", "")
                            for clip in clips:
                                annotations = clip.get("annotations", [])
                                clip_uid = clip.get("clip_uid", "")
                                for ann in annotations:
                                    labels = ann.get("labels", [ann])
                                    for label in labels:
                                        label["video_uid"] = video_uid
                                        label["clip_uid"] = clip_uid
                                        yield label
                elif isinstance(data, list):
                    for item in data:
                        yield item
            except (json.JSONDecodeError, IOError):
                continue

    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability = SubCapability.EGOCENTRIC_INTENT_NEXT_STEP,
        task_type: TaskType = None,
    ) -> CanonicalSample:
        video_uid = raw.get("video_uid", raw.get("video_id", ""))
        clip_uid = raw.get("clip_uid", "")
        raw_id = raw.get("annotation_uid", raw.get("id", f"{video_uid}_{clip_uid}"))

        # Extract question/answer based on annotation type
        query = raw.get("query", raw.get("label", raw.get("narration_text", "")))
        answer = raw.get("answer", raw.get("response", query))

        # Temporal info
        start_s = raw.get("clip_start_sec", raw.get("start_sec", None))
        end_s = raw.get("clip_end_sec", raw.get("end_sec", None))

        if task_type is None:
            if raw.get("options") or raw.get("choices"):
                task_type = TaskType.MCQ
            elif start_s is not None and end_s is not None:
                task_type = TaskType.GROUNDING
            else:
                task_type = TaskType.OPEN_ENDED

        # Format question based on sub-capability
        if sub_capability == SubCapability.EGOCENTRIC_INTENT_NEXT_STEP:
            question_text = f"Question: In this egocentric video, what is the person doing and what will they do next? Context: {query}"
        elif sub_capability == SubCapability.LONG_VIDEO_RETRIEVAL_MEMORY:
            question_text = f"Question: {query}"
        elif sub_capability == SubCapability.FUTURE_EVENT_COUNTERFACTUAL:
            question_text = f"Question: Based on what you see in this video, what happens next? Context: {query}"
        else:
            question_text = f"Question: {query}"

        video_url = self.video_path(str(video_uid))
        messages = self.make_messages(
            video_url=video_url,
            question_text=question_text,
            answer_text=str(answer),
            start_s=start_s,
            end_s=end_s,
        )

        graders = [self.make_exact_match_grader(str(answer))]

        # Select reward template based on capability
        reward_map = {
            SubCapability.EGOCENTRIC_INTENT_NEXT_STEP: (
                "ego_intent_v1",
                {"what": 0.35, "why": 0.30, "next": 0.25, "format": 0.10}
            ),
            SubCapability.LONG_VIDEO_RETRIEVAL_MEMORY: (
                "long_retrieval_v1",
                {"ans": 0.35, "retrieval": 0.35, "support": 0.20, "efficiency": 0.10}
            ),
            SubCapability.FUTURE_EVENT_COUNTERFACTUAL: (
                "future_pred_v1",
                {"ans": 0.5, "future_slot": 0.2, "contrast": 0.2, "format": 0.1}
            ),
            SubCapability.MULTIMODAL_AV_FUSION: (
                "av_fusion_v1",
                {"ans": 0.55, "cross_modal": 0.20, "ablation_consistency": 0.15, "format": 0.10}
            ),
        }
        template, weights = reward_map.get(
            sub_capability,
            ("ego_intent_v1", {"what": 0.35, "why": 0.30, "next": 0.25, "format": 0.10})
        )

        return CanonicalSample(
            messages=messages,
            graders=graders,
            data_info=DataInfo(
                data_id=self.make_data_id(str(raw_id)),
                ability="Egocentric",
                datasource=self.display_name,
                sub_ability=[sub_capability.value],
                task_type=task_type,
                modality_profile=ModalityProfile(
                    video=True,
                    audio=sub_capability == SubCapability.MULTIMODAL_AV_FUSION,
                ),
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
                    mix_bucket=sub_capability.value,
                ),
            ),
        )
