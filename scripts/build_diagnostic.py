"""Generate ~50 diagnostic evaluation samples for quick sub-capability probing.

Each sample covers one of the 11 sub-capabilities. The questions are designed
to test specific reasoning patterns, making it easy to detect model weaknesses.

Usage:
    python -m scripts.build_diagnostic [--output data/diagnostic/diagnostic_eval.jsonl]
"""

from __future__ import annotations

import json
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


def _msg(video_url: str, question: str, answer: str, subtitle: Optional[str] = None) -> list:
    """Build standard messages."""
    user_content = [
        VideoURLContent(video_url=VideoURL(url=video_url)),
        TextContent(text=question),
    ]
    if subtitle:
        user_content.append(TextContent(text=f"Subtitle context: {subtitle}"))
    return [
        Message(role="system", content=[TextContent(text="You are a video reasoning assistant. Follow the required answer format exactly.")]),
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
            datasource="diagnostic_synthetic",
            special_purpose="diagnostic_eval",
            sub_ability=sub,
            task_type=task,
            split="eval",
            evidence=evidence,
            build_info=BuildInfo(build_type=BuildType.SYNTHETIC),
        ),
        extra_info=ExtraInfo(
            sampling_info=SamplingInfo(mix_bucket=cap),
        ),
    )
    return json.loads(s.model_dump_json())


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic samples: 11 capabilities × ~5 each ≈ 50 samples
# ─────────────────────────────────────────────────────────────────────────────

SAMPLES = []
_id = 0

def _next_id(prefix: str) -> str:
    global _id
    _id += 1
    return f"diag_{prefix}_{_id:03d}"


# ═══════════════════════════════════════════════════════════════════════════════
#  A. temporal_atomic — 事件先后/速度/方向/状态变化
# ═══════════════════════════════════════════════════════════════════════════════

V = "file:///data/diagnostic/placeholder.mp4"

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["event_order"], "mcq", V,
    "Question: In this video, a person first picks up a cup, then sits down, then drinks. Which action happens SECOND? Options: A. pick up cup B. sit down C. drink D. stand up. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["before_after"], "mcq", V,
    "Question: A ball is thrown upward, reaches the peak, then falls down. What happens immediately BEFORE the ball falls down? Options: A. thrown upward B. reaches the peak C. bounces D. rolls. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["speed"], "mcq", V,
    "Question: A car accelerates from 0 to 60 km/h in 5 seconds, then maintains speed for 10 seconds, then decelerates to 0 in 3 seconds. During which phase is the car moving fastest? Options: A. acceleration phase B. constant speed phase C. deceleration phase D. all the same. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["attribute_change"], "mcq", V,
    "Question: A traffic light changes from green to yellow to red. After the light turns yellow, what color will it change to NEXT? Options: A. green B. yellow C. red D. blue. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("ta"), "temporal_atomic", "Temporal", ["action_phase"], "mcq", V,
    "Question: A gymnast performs a vault: runs, jumps on the springboard, pushes off the vault table, flips in the air, and lands. What phase comes directly after pushing off the vault table? Options: A. running B. jumping on springboard C. flipping in the air D. landing. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  B. temporal_count_order — 计数/排序
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_count"], "count", V,
    "Question: In the video, a person claps 3 times, pauses, then claps 2 more times. How many total claps are there? Answer with a number only.",
    "5", _exact_grader("5"),
))

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_order"], "order", V,
    "Question: A chef performs these actions in the video: chops onion, boils water, adds salt, stirs the pot. What is the correct order? Options: A. chops→boils→adds salt→stirs B. boils→chops→adds salt→stirs C. chops→adds salt→boils→stirs D. stirs→boils→adds salt→chops. Answer with one capital letter.",
    "A", _mcq_grader("A"),
))

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["repeated_events"], "count", V,
    "Question: A dog fetches a ball. The owner throws the ball 4 times. The dog successfully retrieves it 3 times and misses once. How many successful retrievals? Answer with a number only.",
    "3", _exact_grader("3"),
))

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["step_indexing"], "mcq", V,
    "Question: In an assembly tutorial, the steps are: (1) attach base, (2) insert screws, (3) mount panel, (4) connect wires, (5) test power. Which step number involves 'connect wires'? Options: A. 2 B. 3 C. 4 D. 5. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  C. temporal_grounding — 时序定位
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["moment_retrieval"], "grounding", V,
    "Question: The video is 60 seconds long. A person starts dancing at 15s and stops at 30s. Locate the moment when the person is dancing. Answer with start and end timestamps in seconds, e.g., [start, end].",
    "[15.0, 30.0]", _span_grader("[15.0, 30.0]"),
    evidence=Evidence(support_spans=[[15.0, 30.0]]),
))

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["moment_retrieval"], "grounding", V,
    "Question: In a 120-second cooking video, the chef adds spices between 45s and 55s, then plates the dish between 90s and 110s. When does the chef plate the dish? Answer with [start, end].",
    "[90.0, 110.0]", _span_grader("[90.0, 110.0]"),
    evidence=Evidence(support_spans=[[90.0, 110.0]]),
))

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["highlight_detection"], "grounding", V,
    "Question: In a 90-second soccer video, goals are scored at 20-22s and 75-78s. Locate ALL goal-scoring moments. Answer as [start1, end1]; [start2, end2].",
    "[20.0, 22.0]; [75.0, 78.0]", _span_grader("[20.0, 22.0]; [75.0, 78.0]"),
    evidence=Evidence(support_spans=[[20.0, 22.0], [75.0, 78.0]]),
))

SAMPLES.append(_sample(
    _next_id("tg"), "temporal_grounding", "Temporal", ["support_span_localization"], "grounding", V,
    "Question: A 45-second video shows a cat sleeping (0-20s), waking up (20-25s), and playing with a toy (25-45s). When does the cat wake up? Answer with [start, end].",
    "[20.0, 25.0]", _span_grader("[20.0, 25.0]"),
    evidence=Evidence(support_spans=[[20.0, 25.0]]),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  D. long_video_retrieval_memory — 长视频记忆检索
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["needle_retrieval"], "mcq", V,
    "Question: In a 30-minute lecture video, the professor mentions a key formula 'E=mc²' exactly once at around the 18-minute mark. What formula did the professor mention? Options: A. F=ma B. E=mc² C. PV=nRT D. a²+b²=c². Answer with one capital letter.",
    "B", _mcq_grader("B"),
    subtitle="...and as Einstein showed, E equals m c squared, which fundamentally changed physics...",
))

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["referred_reasoning"], "mcq", V,
    "Question: In a 1-hour documentary, the narrator explains at minute 5 that coral reefs need water above 20°C, and at minute 45 mentions a reef in water at 15°C. Is this second reef likely healthy? Options: A. Yes, it's thriving B. No, the water is too cold C. Cannot determine D. Temperature doesn't matter. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["subtitle_grounded_lookup"], "mcq", V,
    "Question: Based on the subtitles, Character A says 'I'll meet you at the park at 3pm' in scene 2, and in scene 7, Character B asks 'Where should we go?' What should Character B's answer be? Options: A. the mall B. the park C. the office D. home. Answer with one capital letter.",
    "B", _mcq_grader("B"),
    subtitle="Scene 2 - A: 'I'll meet you at the park at 3pm.' ... Scene 7 - B: 'Where should we go?'",
))

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["long_context_memory"], "mcq", V,
    "Question: A security camera records 24 hours. A red car parks at 8:15 AM, a blue truck arrives at 2:30 PM, and the red car leaves at 6:45 PM. How long was the red car parked? Options: A. ~6.5 hours B. ~8 hours C. ~10.5 hours D. ~4 hours. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  E. causal_relation_reasoning — 因果推理
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["causal_why_how"], "mcq", V,
    "Question: In the video, a child pushes a tower of blocks and it falls over. Why did the tower fall? Options: A. The blocks were glued B. The child pushed it C. The wind blew D. It was already falling. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["feasibility"], "mcq", V,
    "Question: A person tries to pour water from an upside-down sealed bottle. Will water come out? Options: A. Yes, immediately B. No, the seal prevents it C. Only if shaken D. Yes, due to gravity. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["interaction_logic"], "mcq", V,
    "Question: Person A hands a book to Person B. Person B places it on a table. Person C picks up the book from the table. Who last had physical contact with the book before Person C? Options: A. Person A B. Person B C. The table D. Nobody. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["temporal_relation"], "mcq", V,
    "Question: It starts raining. Then the ground gets wet. Then a person slips on the wet ground. What is the root cause of the person slipping? Options: A. The ground being wet B. The rain C. The person walking D. Gravity. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("cr"), "causal_relation_reasoning", "Causal", ["causal_why_how"], "mcq", V,
    "Question: A glass of ice water is left in the sun. After 30 minutes, the ice has melted. Why did the ice melt? Options: A. The glass broke B. Someone stirred it C. Heat from sunlight D. The water was already warm. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  F. future_event_counterfactual — 未来预测/反事实
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["what_happens_next"], "mcq", V,
    "Question: A person is holding a full glass of water and trips on a step. What most likely happens next? Options: A. The water stays in the glass B. The water spills C. The glass flies upward D. Nothing happens. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["counterfactual_outcome"], "mcq", V,
    "Question: In the video, a goalkeeper dives left and catches the ball. If the goalkeeper had dived right instead, what would have happened? Options: A. Still caught the ball B. The ball would have gone in C. The game would stop D. The ball would disappear. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["anticipation"], "mcq", V,
    "Question: Dark clouds are forming, the wind is picking up, and people on the street are opening umbrellas. What is about to happen? Options: A. An earthquake B. It will rain C. A parade is coming D. The sun will come out. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["future_event_summary"], "mcq", V,
    "Question: A pot of water is on the stove with the burner on high. Bubbles are starting to form at the bottom. What will happen in the next minute? Options: A. The water will freeze B. The water will boil fully C. The pot will explode D. The water will evaporate instantly. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  G. egocentric_intent_next_step — 自中心意图/下一步
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["current_step_intent"], "mcq", V,
    "Question: In the egocentric video, the person's hands are holding a knife and a carrot on a cutting board. What is the person's current intent? Options: A. Washing dishes B. Cutting the carrot C. Eating the carrot D. Throwing away the carrot. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["next_step_planning"], "mcq", V,
    "Question: The person has just cracked eggs into a bowl and is now reaching for a whisk. What is the most likely next step? Options: A. Crack more eggs B. Beat the eggs C. Pour the eggs into a pan D. Add flour. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["higher_level_goal"], "mcq", V,
    "Question: Across the video, the person gathers flour, sugar, eggs, and butter, preheats the oven, and mixes ingredients in a bowl. What is the person's overall goal? Options: A. Cleaning the kitchen B. Baking a cake C. Making a salad D. Doing inventory. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("ei"), "egocentric_intent_next_step", "Egocentric", ["procedural_anticipation"], "mcq", V,
    "Question: In an egocentric repair video, the person has removed a flat tire and is now positioning the spare tire. What step comes after mounting the spare? Options: A. Remove the flat tire B. Tighten the lug nuts C. Lower the car D. Drive away immediately. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  H. spatial_spatiotemporal_reasoning — 空间/时空推理
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["relative_position"], "mcq", V,
    "Question: In the scene, a red ball is on top of a blue box, and the blue box is to the left of a green cylinder. Where is the red ball relative to the green cylinder? Options: A. To the right B. To the upper-left C. Below D. Behind. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["motion_in_space"], "mcq", V,
    "Question: A car starts at the left side of the frame, moves to the right, then makes a U-turn and drives back to the left. Where does the car end up? Options: A. Right side B. Center C. Left side D. Off-screen. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["object_person_relation"], "mcq", V,
    "Question: Person A is standing between a table and a door. Person B is behind the table. Person A moves toward the door. Now who is closer to the table? Options: A. Person A B. Person B C. They are equidistant D. Neither is close. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["topological_change"], "mcq", V,
    "Question: A ring is threaded onto a rope. The rope is then cut in the middle. How many separate pieces of rope are there, and is the ring still on a rope? Options: A. 2 pieces, ring still on one B. 2 pieces, ring falls off C. 1 piece, ring still on D. 3 pieces. Answer with one capital letter.",
    "A", _mcq_grader("A"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  I. multimodal_av_fusion — 音视频融合
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["sound_source_alignment"], "mcq", V,
    "Question: In the video, a dog is visible on the left and a cat on the right. A barking sound is heard. Which animal is making the sound? Options: A. The cat B. The dog C. Neither D. Both. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["cross_modal_disambiguation"], "mcq", V,
    "Question: The video shows a person moving their lips, but the audio says 'I love cats' while the subtitles read 'I love bats'. Based on the audio, what did the person actually say? Options: A. I love cats B. I love bats C. I love hats D. I love rats. Answer with one capital letter.",
    "A", _mcq_grader("A"),
))

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["subtitle_audio_video_fusion"], "mcq", V,
    "Question: A news video shows a reporter at a fire scene. The audio has sirens. The subtitle reads 'Three fire trucks responded.' How many fire trucks responded according to the subtitle? Options: A. 1 B. 2 C. 3 D. 4. Answer with one capital letter.",
    "C", _mcq_grader("C"),
    subtitle="Three fire trucks responded to the blaze in downtown.",
))

SAMPLES.append(_sample(
    _next_id("av"), "multimodal_av_fusion", "AV-Fusion", ["cross_modal_disambiguation"], "mcq", V,
    "Question: A musician is shown playing a guitar, but the audio is of a piano. What instrument is ACTUALLY being heard? Options: A. Guitar B. Piano C. Drums D. Violin. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  J. topic_plot_knowledge_acquisition — 主题/知识获取
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["tutorial_topic_understanding"], "mcq", V,
    "Question: A tutorial video demonstrates how to tie a bowline knot: make a loop, pass the end through, go around the standing line, and back through the loop. What knot is being taught? Options: A. Square knot B. Bowline knot C. Clove hitch D. Slip knot. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["plot_theme"], "mcq", V,
    "Question: In a short film, a lonely old man finds a stray dog, cares for it, and gradually reconnects with his neighbors through the dog. What is the main theme? Options: A. Animal training B. Loneliness and connection C. Financial hardship D. Adventure. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["knowledge_acquisition"], "mcq", V,
    "Question: A science video explains that photosynthesis converts CO₂ and H₂O into glucose and O₂ using sunlight. What is a product of photosynthesis? Options: A. Carbon dioxide B. Water C. Glucose D. Nitrogen. Answer with one capital letter.",
    "C", _mcq_grader("C"),
))

SAMPLES.append(_sample(
    _next_id("tk"), "topic_plot_knowledge_acquisition", "Knowledge", ["adaptation_after_watching"], "mcq", V,
    "Question: After watching a tutorial on CPR: 30 chest compressions then 2 rescue breaths, repeated. If someone collapses, what should you do first according to the video? Options: A. Give rescue breaths B. Call for help and start chest compressions C. Move them to a hospital D. Give water. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

# ═══════════════════════════════════════════════════════════════════════════════
#  K. anti_shortcut_contrast — 反捷径/对比鲁棒性
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["reverse_order"], "mcq", V,
    "Question: [REVERSED VIDEO] In this reversed video, a person is seen un-sitting (rising from a chair) and then un-entering a room (walking backward out). In the ORIGINAL (non-reversed) video, what did the person do first? Options: A. Sat down B. Entered the room C. Left the room D. Stood up. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["last_frame_only"], "mcq", V,
    "Question: If you could only see the LAST FRAME of a video showing a completed jigsaw puzzle, could you determine the order in which pieces were placed? Options: A. Yes, from the last frame B. No, the last frame only shows the final state C. Yes, from color patterns D. Yes, from the edges. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["sparse_frame_ablation"], "mcq", V,
    "Question: A 10-second video is sampled at only 2 frames (first and last). The first frame shows an empty table. The last frame shows a vase on the table. Can you determine WHO placed the vase? Options: A. Yes B. No, sparse sampling loses this info C. Yes, from the vase style D. Yes, from the table. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("as"), "anti_shortcut_contrast", "Robustness", ["subtitle_drop"], "mcq", V,
    "Question: A video originally has subtitles saying 'The answer is Paris.' With subtitles removed, and only visual/audio cues of the Eiffel Tower visible, what city is shown? Options: A. London B. Paris C. Rome D. Berlin. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))


# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
#  Extra 4 samples to reach exactly 50
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES.append(_sample(
    _next_id("tco"), "temporal_count_order", "Temporal", ["action_count"], "count", V,
    "Question: In a workout video, the person does 12 push-ups, rests, then does 8 more push-ups. How many push-ups total? Answer with a number only.",
    "20", _exact_grader("20"),
))

SAMPLES.append(_sample(
    _next_id("lv"), "long_video_retrieval_memory", "Memory", ["needle_retrieval"], "mcq", V,
    "Question: In a 45-minute nature documentary, the narrator mentions the name of a rare bird 'Kakapo' exactly once, around minute 32. What is the name of the rare bird? Options: A. Dodo B. Kakapo C. Kiwi D. Condor. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("fc"), "future_event_counterfactual", "Prediction", ["anticipation"], "mcq", V,
    "Question: A stack of dominos is set up in a line. A person flicks the first domino. What will happen? Options: A. Only the first falls B. They all fall in sequence C. They all fall simultaneously D. Nothing happens. Answer with one capital letter.",
    "B", _mcq_grader("B"),
))

SAMPLES.append(_sample(
    _next_id("sp"), "spatial_spatiotemporal_reasoning", "Spatial", ["motion_in_space"], "mcq", V,
    "Question: A drone flies north for 100m, turns east for 50m, then turns south for 100m. Relative to the start, where is the drone now? Options: A. 50m east B. 50m north C. 100m northeast D. Back at start. Answer with one capital letter.",
    "A", _mcq_grader("A"),
))


def build_diagnostic(output: str = "data/diagnostic/diagnostic_eval.jsonl"):
    """Write diagnostic samples to JSONL."""
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
    print("\nPer-capability breakdown:")
    for cap, cnt in sorted(caps.items()):
        print(f"  {cap}: {cnt}")


if __name__ == "__main__":
    build_diagnostic()
