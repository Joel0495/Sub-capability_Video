"""Generate small synthetic diagnostic videos using ffmpeg.

Each video is a few seconds long, low-resolution (320x240), and shows specific
visual content needed by the diagnostic evaluation questions.

Usage:
    python -m scripts.generate_diagnostic_videos [--output_dir data/diagnostic/videos]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path("data/diagnostic/videos")


def run_ffmpeg(args: list, name: str):
    """Run ffmpeg with given args, print status."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    print(f"  Generating {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
        return False
    return True


def generate_all(output_dir: str = "data/diagnostic/videos"):
    """Generate all diagnostic videos."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating diagnostic videos in {out}/\n")

    # ─── Video 1: Three colored circles appearing in sequence (red, blue, green) ───
    # Tests: temporal_atomic (event order), temporal_count_order (counting)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=black:s=320x240:d=6,drawtext=text='':fontcolor=white:fontsize=1,"
            # Red circle appears 0-6s (left)
            "drawbox=x=40:y=80:w=80:h=80:color=red@1.0:t=fill:enable='gte(t,0.5)',"
            # Blue circle appears 2-6s (center)
            "drawbox=x=120:y=80:w=80:h=80:color=blue@1.0:t=fill:enable='gte(t,2.0)',"
            # Green circle appears 4-6s (right)
            "drawbox=x=200:y=80:w=80:h=80:color=green@1.0:t=fill:enable='gte(t,4.0)'"
        ),
        "-t", "6", "-r", "10", str(out / "v01_three_objects_appear.mp4")
    ], "v01: three objects appearing in sequence")

    # ─── Video 2: Object moving left to right then back ───
    # Tests: spatial (motion direction), temporal_atomic (direction change)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=white:s=320x240:d=6,"
            # Yellow box moves right (0-3s) then left (3-6s)
            "drawbox=x='if(lt(t,3),40+80*t/3,40+80-80*(t-3)/3)':y=100:w=40:h=40"
            ":color=yellow@1.0:t=fill"
        ),
        "-t", "6", "-r", "10", str(out / "v02_object_moves_right_left.mp4")
    ], "v02: object moving right then left")

    # ─── Video 3: Two objects collide in the middle ───
    # Tests: causal_relation, future_event_counterfactual
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=black:s=320x240:d=4,"
            # Red box moves from left to center
            "drawbox=x='min(20+60*t,140)':y=100:w=40:h=40:color=red@1.0:t=fill,"
            # Blue box moves from right to center
            "drawbox=x='max(260-60*t,140)':y=100:w=40:h=40:color=blue@1.0:t=fill"
        ),
        "-t", "4", "-r", "10", str(out / "v03_two_objects_collide.mp4")
    ], "v03: red and blue objects collide")

    # ─── Video 4: Counting objects - 5 boxes appear one by one ───
    # Tests: temporal_count_order (counting)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=gray:s=320x240:d=5,"
            "drawbox=x=10:y=100:w=40:h=40:color=red@1.0:t=fill:enable='gte(t,0.5)',"
            "drawbox=x=70:y=100:w=40:h=40:color=blue@1.0:t=fill:enable='gte(t,1.5)',"
            "drawbox=x=130:y=100:w=40:h=40:color=green@1.0:t=fill:enable='gte(t,2.5)',"
            "drawbox=x=190:y=100:w=40:h=40:color=yellow@1.0:t=fill:enable='gte(t,3.5)',"
            "drawbox=x=250:y=100:w=40:h=40:color=white@1.0:t=fill:enable='gte(t,4.5)'"
        ),
        "-t", "5", "-r", "10", str(out / "v04_five_objects_counting.mp4")
    ], "v04: five objects appear for counting")

    # ─── Video 5: Object falling down (gravity simulation) ───
    # Tests: causal_relation (why it falls), spatial (motion direction)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=lightblue:s=320x240:d=3,"
            # Red box falls from top to bottom (accelerating)
            "drawbox=x=140:y='min(10+80*t*t,200)':w=40:h=40:color=red@1.0:t=fill,"
            # Ground line
            "drawbox=x=0:y=220:w=320:h=20:color=brown@1.0:t=fill"
        ),
        "-t", "3", "-r", "15", str(out / "v05_object_falling.mp4")
    ], "v05: object falling down")

    # ─── Video 6: Color change sequence (red -> yellow -> green like traffic light) ───
    # Tests: temporal_atomic (attribute change), temporal_count_order (order)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=black:s=320x240:d=6,"
            # Full-screen color changes
            "drawbox=x=110:y=20:w=100:h=100:color=red@1.0:t=fill:enable='lt(t,2)',"
            "drawbox=x=110:y=20:w=100:h=100:color=yellow@1.0:t=fill:enable='between(t,2,4)',"
            "drawbox=x=110:y=20:w=100:h=100:color=green@1.0:t=fill:enable='gte(t,4)',"
            # Label at bottom
            "drawbox=x=0:y=200:w=320:h=40:color=0x333333@1.0:t=fill"
        ),
        "-t", "6", "-r", "10", str(out / "v06_color_change_sequence.mp4")
    ], "v06: color change (traffic light pattern)")

    # ─── Video 7: Object appears, stays, disappears ───
    # Tests: temporal_grounding (when object is visible)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=white:s=320x240:d=10,"
            # Blue box visible only between 3s and 7s
            "drawbox=x=120:y=80:w=80:h=80:color=blue@1.0:t=fill:enable='between(t,3,7)',"
            # Timer bar at bottom
            "drawbox=x=0:y=220:w='320*t/10':h=20:color=0x444444@1.0:t=fill"
        ),
        "-t", "10", "-r", "10", str(out / "v07_object_appears_disappears.mp4")
    ], "v07: object appears at 3s, disappears at 7s")

    # ─── Video 8: Two objects, different sizes (spatial relationship) ───
    # Tests: spatial (relative position, size comparison)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=black:s=320x240:d=4,"
            # Large red box on the left
            "drawbox=x=30:y=50:w=100:h=100:color=red@1.0:t=fill,"
            # Small green box on the right, higher up
            "drawbox=x=220:y=30:w=50:h=50:color=green@1.0:t=fill,"
            # Small blue box at bottom-right
            "drawbox=x=200:y=160:w=60:h=60:color=blue@1.0:t=fill"
        ),
        "-t", "4", "-r", "10", str(out / "v08_spatial_layout.mp4")
    ], "v08: spatial layout (3 objects, different positions/sizes)")

    # ─── Video 9: Fast object vs slow object ───
    # Tests: temporal_atomic (speed comparison)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=white:s=320x240:d=5,"
            # Fast red box (top row, crosses screen in 2.5s)
            "drawbox=x='min(10+120*t,280)':y=40:w=30:h=30:color=red@1.0:t=fill,"
            # Slow blue box (bottom row, crosses in 5s)
            "drawbox=x='10+54*t':y=160:w=30:h=30:color=blue@1.0:t=fill"
        ),
        "-t", "5", "-r", "10", str(out / "v09_fast_vs_slow.mp4")
    ], "v09: fast red vs slow blue object")

    # ─── Video 10: Object bouncing (up-down-up) ───
    # Tests: temporal_atomic (phase detection), future prediction
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=0xCCCCCC:s=320x240:d=6,"
            # Bouncing ball: goes down 0-2s, up 2-4s, down 4-6s
            "drawbox=x=140:y='if(lt(t,2),20+90*t,if(lt(t,4),200-90*(t-2),20+90*(t-4)))'"
            ":w=30:h=30:color=red@1.0:t=fill,"
            # Ground
            "drawbox=x=0:y=220:w=320:h=20:color=0x444444@1.0:t=fill"
        ),
        "-t", "6", "-r", "15", str(out / "v10_bouncing_object.mp4")
    ], "v10: bouncing object")

    # ─── Video 11: Screen split - left side dark, right side bright ───
    # Tests: spatial reasoning, contrast detection
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=black:s=320x240:d=4,"
            # Right half is white
            "drawbox=x=160:y=0:w=160:h=240:color=white@1.0:t=fill,"
            # Red object on the dark side
            "drawbox=x=50:y=100:w=40:h=40:color=red@1.0:t=fill,"
            # Green object on the bright side
            "drawbox=x=230:y=100:w=40:h=40:color=green@1.0:t=fill"
        ),
        "-t", "4", "-r", "10", str(out / "v11_split_screen.mp4")
    ], "v11: split screen (dark left, bright right)")

    # ─── Video 12: Objects appearing at different positions over time ───
    # Tests: temporal_grounding, long video retrieval
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=0x222222:s=320x240:d=12,"
            # Red box at 1-3s top-left
            "drawbox=x=20:y=20:w=60:h=60:color=red@1.0:t=fill:enable='between(t,1,3)',"
            # Blue box at 4-6s center
            "drawbox=x=130:y=90:w=60:h=60:color=blue@1.0:t=fill:enable='between(t,4,6)',"
            # Green box at 7-9s bottom-right
            "drawbox=x=240:y=160:w=60:h=60:color=green@1.0:t=fill:enable='between(t,7,9)',"
            # Yellow box at 10-12s top-right
            "drawbox=x=240:y=20:w=60:h=60:color=yellow@1.0:t=fill:enable='between(t,10,12)'"
        ),
        "-t", "12", "-r", "10", str(out / "v12_sequential_positions.mp4")
    ], "v12: objects at different positions over time")

    # ─── Video 13: Reversed motion (object moves right to left) ───
    # Tests: anti_shortcut (reverse order detection)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=white:s=320x240:d=4,"
            # Object moves from right to left (reversed motion)
            "drawbox=x='280-70*t':y=100:w=30:h=30:color=red@1.0:t=fill"
        ),
        "-t", "4", "-r", "10", str(out / "v13_reverse_motion.mp4")
    ], "v13: object moving right to left")

    # ─── Video 14: Multiple events in a longer clip ───
    # Tests: long_video_retrieval, needle retrieval
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=black:s=320x240:d=15,"
            # Background objects that persist
            "drawbox=x=10:y=200:w=300:h=30:color=0x333333@1.0:t=fill,"
            # Brief red flash at 3s (blink)
            "drawbox=x=0:y=0:w=320:h=240:color=red@0.5:t=fill:enable='between(t,3,3.5)',"
            # Green box appears at 7s (event)
            "drawbox=x=140:y=100:w=40:h=40:color=green@1.0:t=fill:enable='between(t,7,8)',"
            # Blue flash at 11s
            "drawbox=x=0:y=0:w=320:h=240:color=blue@0.5:t=fill:enable='between(t,11,11.5)',"
            # White box at 13s
            "drawbox=x=60:y=60:w=40:h=40:color=white@1.0:t=fill:enable='between(t,13,14)'"
        ),
        "-t", "15", "-r", "10", str(out / "v14_multiple_events_long.mp4")
    ], "v14: multiple events in long clip")

    # ─── Video 15: Stacking/overlay (objects on top of each other) ───
    # Tests: spatial (topological relation, occlusion)
    run_ffmpeg([
        "-f", "lavfi", "-i",
        (
            "color=c=white:s=320x240:d=4,"
            # Large blue box (bottom)
            "drawbox=x=100:y=100:w=120:h=100:color=blue@1.0:t=fill,"
            # Medium red box (overlapping, shifted right-up)
            "drawbox=x=140:y=70:w=100:h=80:color=red@1.0:t=fill,"
            # Small yellow box (on top of both)
            "drawbox=x=170:y=50:w=50:h=50:color=yellow@1.0:t=fill"
        ),
        "-t", "4", "-r", "10", str(out / "v15_stacking_overlap.mp4")
    ], "v15: overlapping objects (spatial layering)")

    # Verify outputs
    print(f"\n{'='*60}")
    count = 0
    total_size = 0
    for f in sorted(out.glob("*.mp4")):
        size_kb = f.stat().st_size / 1024
        total_size += size_kb
        count += 1
        print(f"  {f.name:45s} {size_kb:6.1f} KB")
    print(f"{'='*60}")
    print(f"  Total: {count} videos, {total_size:.1f} KB ({total_size/1024:.2f} MB)")


if __name__ == "__main__":
    generate_all()
