"""Generate ~50 diagnostic evaluation samples using real synthetic videos.

Each sample covers one of the 11 sub-capabilities. Questions are designed so
that models MUST watch the video to answer correctly — text alone is ambiguous.

Pre-requisite: run `python -m scripts.generate_diagnostic_videos` first to
create the 15 synthetic videos in data/diagnostic/videos/.

Usage:
    python -m scripts.build_diagnostic [--output data/diagnostic/diagnostic_eval.jsonl]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema.canonical import (
    CanonicalSample,
    SubCapability,
    TaskType,
    BuildType,
    VideoURL,
    TextContent,
    VideoURLContent,
    Message,
    Grader,
    GraderParams,
    DataInfo,
    BuildInfo,
    Evidence,
    ExtraInfo,
    SamplingInfo,
)

# Base path for diagnostic videos (absolute, will be set at build time)
_VIDEO_BASE = Path(__file__).resolve().parent.parent / "data" / "diagnostic" / "videos"


def _vpath(filename: str) -> str:
    """Return file:// URL for a diagnostic video."""
    return f"file://{_VIDEO_BASE / filename}"


def _msg(video_url: str, question: str, answer: str, subtitle: Optional[str] = None) -> list:
    """Build standard messages."""
    user_content = [
        VideoURLContent(video_url=VideoURL(url=video_url)),
        TextContent(text=question),
    ]
    if subtitle:
        user_content.append(TextContent(text=f"Subtitle context: {subtitle}"))
    return [
        Message(role="system", content=[TextContent(
            text="You are a video reasoning assistant. Watch the video carefully "
                 "and answer based ONLY on what you observe. Follow the required answer format exactly."
        )]),
        Message(role="user", content=user_content),
        Message(role="assistant", content=[TextContent(text=answer)]),
    ]


def _mcq_grader(gt: str) -> list:
    return [Grader(type="ruler", name="option_match", gt=gt, params=GraderParams(min_score=1.0))]


def _exact_grader(gt: str) -> list:
    return [Grader(type="ruler", name="exact_match", gt=gt, params=GraderParams(min_score=1.0))]


def _span_grader(gt: str) -> list:
    return [Grader(type="ruler", name="span_iou", gt=gt, params=GraderParams(min_score=0.5))]


def _sample(data_id: str, cap: str, ability: str, sub: list, task: str,
            video: str, q: str, a: str, graders: list,
            subtitle: Optional[str] = None, evidence: Optional[Evidence] = None) -> dict:
    s = CanonicalSample(
        messages=_msg(video, q, a, subtitle),
        graders=graders,
        data_info=DataInfo(
            data_id=data_id,
            ability=ability,
            datasource="diagnostic_synthetic_video",
            special_purpose="diagnostic_eval",
            sub_ability=sub,
            task_type=task,
            split="eval",
            evidence=evidence,
            build_info=BuildInfo(
                build_type=BuildType.SYNTHETIC,
                video_path=video,
            ),
        ),
        extra_info=ExtraInfo(
            sampling_info=SamplingInfo(mix_bucket=cap),
        ),
    )
    return json.loads(s.model_dump_json())


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic samples: 11 capabilities × ~4-5 each ≈ 50 samples
#  ALL questions require watching the actual video to answer correctly.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLES = []
_id = 0


def _next_id(prefix: str) -> str:
    global _id
    _id += 1
    return f"diag_{prefix}_{_id:03d}"


# ═══════════════════════════════════════════════════════════════════════════════
#  A. temporal_atomic — event order / speed / direction / state change
#  Videos: v01 (three objects appear), v06 (color change), v09 (fast vs slow)
# ═══════════════════════════════════════════════════════════════════════════════

V01 = _vpath("v01_three_objects_appear.mp4")
V06 = _vpath("v06_color_change_sequence.mp4")
V09 = _vpath("v09_fast_vs_slow.mp4")

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["event_order"], "mcq", V01,
    "Question: Watch the video. Three colored objects appear at different times. "
    "Which color appears SECOND? "
    "Options: A. Red B. Blue C. Green D. Yellow. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["event_order"], "mcq", V01,
    "Question: Watch the video carefully. Which colored object appears FIRST? "
    "Options: A. Green B. Yellow C. Blue D. Red. Answer with one capital letter.",
    "D", _mcq_grader("D"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["attribute_change"], "mcq", V06,
    "Question: Watch the color changes in this video. What color appears BETWEEN "
    "red and green? "
    "Options: A. Blue B. White C. Yellow D. Purple. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["speed"], "mcq", V09,
    "Question: This video shows two objects moving at different speeds. "
    "Which colored object moves FASTER? "
    "Options: A. Blue B. Red C. They move at the same speed D. Green. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["attribute_change"], "mcq", V06,
    "Question: Watch the video. What is the LAST color shown in the sequence? "
    "Options: A. Red B. Blue C. Yellow D. Green. Answer with one capital letter.",
    "D", _mcq_grader("D"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  B. temporal_count_order — counting / ordering
#  Videos: v04 (five objects), v01 (three objects), v12 (sequential positions)
# ═══════════════════════════════════════════════════════════════════════════════

V04 = _vpath("v04_five_objects_counting.mp4")
V12 = _vpath("v12_sequential_positions.mp4")

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_count"], "count", V04,
    "Question: Watch the video and count: How many colored objects appear in total "
    "by the end of the video? Answer with a number only.",
    "5", _exact_grader("5"),
))

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_order"], "mcq", V01,
    "Question: Watch the video. Three objects appear in sequence. "
    "What is the correct order of colors? "
    "Options: A. Red→Blue→Green B. Blue→Red→Green C. Green→Blue→Red D. Red→Green→Blue. "
    "Answer with one capital letter.",
    "A", _mcq_grader("A"),
))

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_count"], "count", V01,
    "Question: Watch the video. How many colored objects appear during the entire video? "
    "Answer with a number only.",
    "3", _exact_grader("3"),
))

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_order"], "mcq", V12,
    "Question: Watch the video. Four colored objects appear at different times and positions. "
    "Which color appears THIRD? "
    "Options: A. Red B. Blue C. Green D. Yellow. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_count"], "count", V12,
    "Question: Watch the entire video. How many different colored objects appear? "
    "Answer with a number only.",
    "4", _exact_grader("4"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  C. temporal_grounding — temporal localization
#  Videos: v07 (object appears/disappears), v12 (sequential positions)
# ═══════════════════════════════════════════════════════════════════════════════

V07 = _vpath("v07_object_appears_disappears.mp4")

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["moment_retrieval"], "grounding", V07,
    "Question: This is a 10-second video. A blue object appears and then disappears. "
    "At approximately what time range is the blue object visible? "
    "Answer with [start, end] in seconds.",
    "[3.0, 7.0]", _span_grader("[3.0, 7.0]"),
    evidence=Evidence(support_spans=[[3.0, 7.0]]),
))

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["moment_retrieval"], "grounding", V12,
    "Question: This is a 12-second video. A red object appears briefly. "
    "At approximately what time range is the red object visible? "
    "Answer with [start, end] in seconds.",
    "[1.0, 3.0]", _span_grader("[1.0, 3.0]"),
    evidence=Evidence(support_spans=[[1.0, 3.0]]),
))

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["moment_retrieval"], "grounding", V12,
    "Question: This is a 12-second video with objects appearing at different times. "
    "When does the blue object appear? Answer with [start, end] in seconds.",
    "[4.0, 6.0]", _span_grader("[4.0, 6.0]"),
    evidence=Evidence(support_spans=[[4.0, 6.0]]),
))

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["highlight_detection"], "mcq", V07,
    "Question: In this 10-second video, a blue object is visible during part of the clip. "
    "For approximately how many seconds is the blue object visible? "
    "Options: A. 2 seconds B. 4 seconds C. 6 seconds D. 8 seconds. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  D. long_video_retrieval_memory — memory / needle retrieval
#  Videos: v14 (multiple events in 15s clip), v12 (sequential positions)
# ═══════════════════════════════════════════════════════════════════════════════

V14 = _vpath("v14_multiple_events_long.mp4")

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["needle_retrieval"], "mcq", V14,
    "Question: Watch the entire 15-second video carefully. "
    "A brief red flash occurs at one point. At approximately what time does the red flash happen? "
    "Options: A. Around 1 second B. Around 3 seconds C. Around 7 seconds D. Around 11 seconds. "
    "Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["needle_retrieval"], "mcq", V14,
    "Question: In this 15-second video, a green object briefly appears. "
    "At approximately what time does it show up? "
    "Options: A. 2 seconds B. 5 seconds C. 7 seconds D. 13 seconds. "
    "Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["referred_reasoning"], "mcq", V14,
    "Question: Watch this video. Two colored flashes occur at different times. "
    "What are their colors? "
    "Options: A. Red and Green B. Red and Blue C. Blue and Green D. Yellow and Red. "
    "Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["long_context_memory"], "count", V14,
    "Question: Watch the entire 15-second video. How many distinct brief events "
    "(flashes or object appearances) occur throughout? Answer with a number only.",
    "4", _exact_grader("4"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  E. causal_relation_reasoning — cause-and-effect
#  Videos: v03 (collision), v05 (falling object)
# ═══════════════════════════════════════════════════════════════════════════════

V03 = _vpath("v03_two_objects_collide.mp4")
V05 = _vpath("v05_object_falling.mp4")

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["causal_why_how"], "mcq", V03,
    "Question: Watch the video. Two objects move toward each other and meet in the middle. "
    "What causes them to meet? "
    "Options: A. They are both stationary B. They both move toward the center "
    "C. Only the red one moves D. Only the blue one moves. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["causal_why_how"], "mcq", V05,
    "Question: Watch the video. A red object moves downward on a light-blue background. "
    "What visual cue suggests the cause of this motion? "
    "Options: A. Another object pushes it B. It accelerates downward (gravity-like) "
    "C. It moves at constant speed D. It moves upward first. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["interaction_logic"], "mcq", V03,
    "Question: In the video, which objects interact with each other? "
    "Options: A. A red and a green object B. A red and a blue object "
    "C. Two blue objects D. A yellow and a red object. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["feasibility"], "mcq", V05,
    "Question: In the video, a red object falls toward a brown surface at the bottom. "
    "What color is the surface it falls toward? "
    "Options: A. White B. Gray C. Brown D. Green. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["temporal_relation"], "mcq", V03,
    "Question: In the video, when the red and blue objects approach each other, "
    "from which side does the red object come? "
    "Options: A. From the right B. From the top C. From the left D. From the bottom. "
    "Answer with one capital letter.",
    "C", _mcq_grader("C"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  F. future_event_counterfactual — prediction / what-if
#  Videos: v10 (bouncing), v03 (collision), v02 (move right then left)
# ═══════════════════════════════════════════════════════════════════════════════

V10 = _vpath("v10_bouncing_object.mp4")
V02 = _vpath("v02_object_moves_right_left.mp4")

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["what_happens_next"], "mcq", V10,
    "Question: Watch the video showing an object bouncing. After the first bounce, "
    "what happens to the object? "
    "Options: A. It stays on the ground B. It moves back upward C. It disappears "
    "D. It moves sideways. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["anticipation"], "mcq", V03,
    "Question: At the beginning of the video, a red object and a blue object "
    "are moving toward each other. What will happen when they reach the center? "
    "Options: A. They will stop and overlap B. They will bounce apart "
    "C. The red one disappears D. They will move to the top. Answer with one capital letter.",
    "A", _mcq_grader("A"),
))

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["counterfactual_outcome"], "mcq", V02,
    "Question: In the video, an object moves right and then comes back left. "
    "If the object had NOT reversed direction, where would it be at the end? "
    "Options: A. Back at the starting position B. Further to the right "
    "C. At the top of the screen D. In the center. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["what_happens_next"], "mcq", V05,
    "Question: In the video a red object falls. Based on its trajectory, "
    "what will happen when it reaches the brown surface? "
    "Options: A. It passes through B. It likely stops or bounces C. It flies upward "
    "D. It turns blue. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  G. egocentric_intent_next_step — intent / next step
#  Videos: v02 (move right-left), v06 (color change), v04 (objects appearing)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["next_step_planning"], "mcq", V04,
    "Question: Watch the video where objects appear one by one. After 4 objects have appeared, "
    "what is the most likely next event based on the observed pattern? "
    "Options: A. All objects disappear B. A 5th object appears C. The video reverses "
    "D. Colors change. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["current_step_intent"], "mcq", V06,
    "Question: The video shows a changing color display. Midway through the video, "
    "what color is currently displayed? "
    "Options: A. Red B. Green C. Yellow D. Blue. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["higher_level_goal"], "mcq", V06,
    "Question: Watch the video. The colors change in a specific pattern: red, yellow, green. "
    "What real-world process does this pattern resemble? "
    "Options: A. A rainbow B. A sunset C. A traffic light D. A painting. "
    "Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["procedural_anticipation"], "mcq", V02,
    "Question: Watch the video. The object moves right and then reverses. "
    "Based on this pattern, if the video continued, what would the object likely do next? "
    "Options: A. Stop permanently B. Move right again C. Move upward D. Disappear. "
    "Answer with one capital letter.",
    "B", _mcq_grader("B"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  H. spatial_spatiotemporal_reasoning — spatial relations
#  Videos: v08 (spatial layout), v11 (split screen), v15 (stacking)
# ═══════════════════════════════════════════════════════════════════════════════

V08 = _vpath("v08_spatial_layout.mp4")
V11 = _vpath("v11_split_screen.mp4")
V15 = _vpath("v15_stacking_overlap.mp4")

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["relative_position"], "mcq", V08,
    "Question: Watch the video. Three colored objects are visible. "
    "Which object is the LARGEST? "
    "Options: A. Red B. Green C. Blue D. They are all the same size. "
    "Answer with one capital letter.",
    "A", _mcq_grader("A"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["relative_position"], "mcq", V08,
    "Question: In the video, which colored object is positioned highest on the screen? "
    "Options: A. Red B. Green C. Blue D. Yellow. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["object_person_relation"], "mcq", V11,
    "Question: The video screen is divided into a dark half and a bright half. "
    "On which side is the red object? "
    "Options: A. The dark (left) side B. The bright (right) side C. In the center "
    "D. At the top. Answer with one capital letter.",
    "A", _mcq_grader("A"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["topological_change"], "mcq", V15,
    "Question: The video shows three overlapping colored objects. "
    "Which object appears on TOP of (in front of) the others? "
    "Options: A. Blue B. Red C. Yellow D. Green. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["relative_position"], "mcq", V11,
    "Question: In the video, what color is the object on the BRIGHT (right) side? "
    "Options: A. Red B. Blue C. Green D. Yellow. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  I. multimodal_av_fusion — audio-visual
#  (Synthetic videos have no audio, so we test visual-text fusion with subtitles)
#  Videos: v11 (split screen), v14 (multiple events)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["cross_modal_disambiguation"], "mcq", V11,
    "Question: The subtitle says 'The bright object is on the left side.' "
    "But watch the video carefully — is this subtitle accurate? "
    "Options: A. Yes, the bright side is on the left B. No, the bright side is on the right "
    "C. There is no bright side D. Both sides are equally bright. Answer with one capital letter.",
    "B", _mcq_grader("B"),
    subtitle="The bright object is on the left side.",
))

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["subtitle_audio_video_fusion"], "mcq", V08,
    "Question: The subtitle claims 'There are two objects in the scene.' "
    "Watch the video. Is this claim correct? "
    "Options: A. Yes, there are exactly 2 B. No, there are 3 C. No, there are 4 "
    "D. No, there is only 1. Answer with one capital letter.",
    "B", _mcq_grader("B"),
    subtitle="There are two objects in the scene.",
))

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["cross_modal_disambiguation"], "mcq", V01,
    "Question: The subtitle says 'The first object to appear is blue.' "
    "Watch the video. Is this subtitle correct? "
    "Options: A. Yes B. No, the first object is red C. No, the first object is green "
    "D. No, the first object is yellow. Answer with one capital letter.",
    "B", _mcq_grader("B"),
    subtitle="The first object to appear is blue.",
))

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["sound_source_alignment"], "mcq", V04,
    "Question: The subtitle states 'Three objects appear in the video.' "
    "Watch the actual video and count. How many objects appear? "
    "Options: A. 3 B. 4 C. 5 D. 6. Answer with one capital letter.",
    "C", _mcq_grader("C"),
    subtitle="Three objects appear in the video.",
))


# ═══════════════════════════════════════════════════════════════════════════════
#  J. topic_plot_knowledge_acquisition — pattern/knowledge extraction
#  Videos: v06 (color change), v04 (objects appearing), v12 (sequential)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["tutorial_topic_understanding"], "mcq", V06,
    "Question: Watch the video. It shows a sequence of color changes. "
    "What pattern do you observe? "
    "Options: A. Random color changes B. Red→Yellow→Green (warm to cool transition) "
    "C. Blue→Green→Red D. All colors appear simultaneously. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["knowledge_acquisition"], "mcq", V04,
    "Question: Watch the video showing objects appearing one by one. "
    "What is the spatial pattern of where they appear? "
    "Options: A. They all appear at the same location B. They appear from left to right "
    "C. They appear from top to bottom D. They appear randomly. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["plot_theme"], "mcq", V12,
    "Question: Watch the video. Objects appear at different positions and times. "
    "What spatial pattern do the object positions follow? "
    "Options: A. All in center B. Moving from top-left toward bottom-right "
    "C. All on the left side D. Alternating sides. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["adaptation_after_watching"], "mcq", V09,
    "Question: After watching this video with two objects moving at different speeds, "
    "which row contains the faster-moving object? "
    "Options: A. Top row B. Bottom row C. They are in the same row "
    "D. There is only one object. Answer with one capital letter.",
    "A", _mcq_grader("A"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  K. anti_shortcut_contrast — robustness / anti-shortcut
#  Videos: v13 (reverse motion), v07 (appears/disappears), v02 (right-left)
# ═══════════════════════════════════════════════════════════════════════════════

V13 = _vpath("v13_reverse_motion.mp4")

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["reverse_order"], "mcq", V13,
    "Question: In this video, an object moves across the screen. "
    "In which direction does it move? "
    "Options: A. Left to right B. Right to left C. Top to bottom D. Bottom to top. "
    "Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["last_frame_only"], "mcq", V01,
    "Question: If you could ONLY see the final frame of this video, could you determine "
    "the ORDER in which the three objects appeared? "
    "Options: A. Yes, from their positions B. No, the final frame shows all three "
    "simultaneously C. Yes, from their colors D. Yes, from their sizes. "
    "Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["sparse_frame_ablation"], "mcq", V07,
    "Question: This 10-second video has a blue object that appears and disappears. "
    "If you only sampled 2 frames (at 0s and 10s), would you see the blue object? "
    "Options: A. Yes, at both frames B. Yes, at one frame C. No, it's gone by 10s "
    "and not yet at 0s D. Yes, it's always there. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["subtitle_drop"], "mcq", V02,
    "Question: Watch the object's movement carefully. "
    "Where does the object END UP at the end of the video? "
    "Options: A. On the right side B. On the left side (where it started) "
    "C. In the center D. Off-screen. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))


# ═══════════════════════════════════════════════════════════════════════════════
#  Extra 2 samples to reach exactly 50
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["moment_retrieval"], "grounding", V12,
    "Question: This is a 12-second video. When does the yellow object appear? "
    "Answer with [start, end] in seconds.",
    "[10.0, 12.0]", _span_grader("[10.0, 12.0]"),
    evidence=Evidence(support_spans=[[10.0, 12.0]]),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["causal_why_how"], "mcq", V09,
    "Question: In this video, one object reaches the right side of the screen before the other. "
    "Why does the red object arrive first? "
    "Options: A. It started further right B. It moves faster C. The blue one stops "
    "D. They arrive at the same time. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

# ─────────────────────────────────────────────────────────────────────────────


def build_diagnostic(output: str = "data/diagnostic/diagnostic_eval.jsonl"):
    """Write diagnostic samples to JSONL."""
    # Verify videos exist
    video_dir = _VIDEO_BASE
    if not video_dir.exists():
        print(f"ERROR: Video directory {video_dir} does not exist.")
        print("Run `python -m scripts.generate_diagnostic_videos` first.")
        sys.exit(1)

    video_count = len(list(video_dir.glob("*.mp4")))
    if video_count < 15:
        print(f"WARNING: Expected 15 videos, found {video_count}.")
        print("Run `python -m scripts.generate_diagnostic_videos` to regenerate.")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in SAMPLES:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Print summary
    from collections import Counter
    caps = Counter(s["extra_info"]["sampling_info"]["mix_bucket"] for s in SAMPLES)
    print(f"Total diagnostic samples: {len(SAMPLES)}")
    print(f"Output: {out_path}")
    print(f"Videos: {video_count} files in {video_dir}/")
    print(f"\nPer-capability breakdown:")
    for cap, cnt in sorted(caps.items()):
        print(f"  {cap}: {cnt}")

    # Verify all video files referenced exist
    missing = []
    for s in SAMPLES:
        for msg in s["messages"]:
            for c in msg["content"]:
                if c.get("type") == "video_url":
                    url = c["video_url"]["url"]
                    fpath = url.replace("file://", "")
                    if not Path(fpath).exists():
                        missing.append(fpath)
    if missing:
        print(f"\nWARNING: {len(missing)} referenced video files are missing!")
        for m in missing[:5]:
            print(f"  {m}")
    else:
        print(f"\nAll referenced video files exist.")


if __name__ == "__main__":
    build_diagnostic()
