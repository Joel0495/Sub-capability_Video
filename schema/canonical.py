"""Pydantic models for the canonical Video RL dataset schema.

This implements the exact JSON structure from the spec:
  top-level keys: messages, graders, data_info, extra_info
  video_url.url: always file:// local paths, never base64
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────


class SubCapability(str, Enum):
    TEMPORAL_ATOMIC = "temporal_atomic"
    TEMPORAL_COUNT_ORDER = "temporal_count_order"
    TEMPORAL_GROUNDING = "temporal_grounding"
    LONG_VIDEO_RETRIEVAL_MEMORY = "long_video_retrieval_memory"
    CAUSAL_RELATION_REASONING = "causal_relation_reasoning"
    FUTURE_EVENT_COUNTERFACTUAL = "future_event_counterfactual"
    EGOCENTRIC_INTENT_NEXT_STEP = "egocentric_intent_next_step"
    SPATIAL_SPATIOTEMPORAL_REASONING = "spatial_spatiotemporal_reasoning"
    MULTIMODAL_AV_FUSION = "multimodal_av_fusion"
    TOPIC_PLOT_KNOWLEDGE_ACQUISITION = "topic_plot_knowledge_acquisition"
    ANTI_SHORTCUT_CONTRAST = "anti_shortcut_contrast"


class TaskType(str, Enum):
    MCQ = "mcq"
    OPEN_ENDED = "open_ended"
    GROUNDING = "grounding"
    COUNT = "count"
    ORDER = "order"
    RETRIEVAL = "retrieval"
    PREFERENCE_PAIR = "preference_pair"
    YES_NO = "yes_no"
    REGRESSION = "regression"


class BuildType(str, Enum):
    NATIVE = "native"
    CONVERTED = "converted"
    SYNTHETIC = "synthetic"
    PAIRED = "paired"


# ─── Video URL ────────────────────────────────────────────────────────────────


class VideoURL(BaseModel):
    url: str = Field(..., description="file:///abs/path or object_store_uri. Never base64.")
    clip_fps: Optional[float] = None
    start_s: Optional[float] = None
    end_s: Optional[float] = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if "base64" in v.lower() or v.startswith("data:"):
            raise ValueError(
                "Videos must be referenced by file:// path or object store URI, "
                "never inlined as base64."
            )
        return v


# ─── Messages (chat format) ──────────────────────────────────────────────────


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class VideoURLContent(BaseModel):
    type: Literal["video_url"] = "video_url"
    video_url: VideoURL


MessageContent = Union[TextContent, VideoURLContent]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[MessageContent]


# ─── Graders ──────────────────────────────────────────────────────────────────


class GraderParams(BaseModel):
    min_score: Optional[float] = None
    threshold: Optional[float] = None
    k: Optional[int] = None
    rubric: Optional[str] = None
    model_config = {"extra": "allow"}


class Grader(BaseModel):
    type: Literal["ruler", "model"] = "ruler"
    name: str
    gt: Optional[str] = None
    params: Optional[GraderParams] = None


# ─── data_info sub-models ────────────────────────────────────────────────────


class ModalityProfile(BaseModel):
    video: bool = True
    audio: bool = False
    subtitle: bool = False
    image_keyframes: bool = False


class VideoProfile(BaseModel):
    video_id: str
    duration_s: Optional[float] = None
    source_fps: Optional[float] = None
    decode_fps: Optional[float] = None
    num_frames: Optional[int] = None
    is_long_video: bool = False


class Evidence(BaseModel):
    support_spans: Optional[list[list[float]]] = None
    distractor_spans: Optional[list[list[float]]] = None
    referred_context: Optional[str] = None


class Perturbation(BaseModel):
    paired_data_id: Optional[str] = None
    type: Optional[str] = None
    answer_changed: Optional[bool] = None
    difficulty: Optional[str] = None


class BuildInfo(BaseModel):
    build_type: BuildType
    builder_version: str = "v1"
    raw_ann_path: Optional[str] = None
    video_path: Optional[str] = None


class DataInfo(BaseModel):
    data_id: str
    domain: str = "Video"
    ability: str
    datasource: str
    special_purpose: str = "rlvr_train"
    sub_ability: list[str] = Field(default_factory=list)
    task_type: TaskType
    split: str = "train"
    license: str = "follow_source_dataset"
    modality_profile: ModalityProfile = Field(default_factory=ModalityProfile)
    video_profile: Optional[VideoProfile] = None
    evidence: Optional[Evidence] = None
    perturbation: Optional[Perturbation] = None
    build_info: BuildInfo


# ─── extra_info sub-models ───────────────────────────────────────────────────


class RewardInfo(BaseModel):
    reward_template: str
    weights: dict[str, float]
    max_reward: float = 1.0
    min_reward: float = 0.0


class SamplingInfo(BaseModel):
    global_weight: float = 1.0
    curriculum_stage: Optional[str] = None
    hard_negative: bool = False
    mix_bucket: str


class QualityInfo(BaseModel):
    video_dependency_score: Optional[float] = None
    text_only_pass_rate: Optional[float] = None
    judge_consistency: Optional[float] = None
    dedup_cluster: Optional[str] = None


class DecodePolicy(BaseModel):
    view_type: str = "global"
    global_fps: Optional[float] = None
    local_windows: list = Field(default_factory=list)
    max_frames: int = 32


class ExtraInfo(BaseModel):
    pass_rate: Optional[float] = None
    pass_rate_model: Optional[str] = None
    pass_rate_list: Optional[list[float]] = None
    pass_rate_samples: Optional[list[str]] = None
    pass_rate_status: Optional[str] = None
    reward_info: Optional[RewardInfo] = None
    sampling_info: Optional[SamplingInfo] = None
    quality_info: Optional[QualityInfo] = None
    decode_policy: Optional[DecodePolicy] = None


# ─── Top-level sample ────────────────────────────────────────────────────────


class CanonicalSample(BaseModel):
    """Top-level schema for every record in the final JSONL.

    Keys: messages, graders, data_info, extra_info
    """
    messages: list[Message]
    graders: list[Grader]
    data_info: DataInfo
    extra_info: Optional[ExtraInfo] = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        roles = [m.role for m in v]
        if "user" not in roles:
            raise ValueError("Messages must contain at least one 'user' message.")
        if "assistant" not in roles:
            raise ValueError("Messages must contain at least one 'assistant' message (ground truth).")
        return v

    @field_validator("graders")
    @classmethod
    def validate_graders(cls, v: list[Grader]) -> list[Grader]:
        if len(v) < 1:
            raise ValueError("At least one grader is required.")
        return v
