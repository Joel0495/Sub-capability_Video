"""Base adapter class for all dataset source adapters.

Each raw dataset gets exactly one adapter. Multiple builders can reuse
the same adapter with different filters for different sub-capabilities.
"""

import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

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
    ModalityProfile,
    VideoProfile,
)


def _load_download_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "download_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class BaseAdapter(ABC):
    """Abstract base adapter for a single dataset source."""

    # Subclasses must set these
    dataset_name: str = ""          # e.g., "nextqa"
    display_name: str = ""          # e.g., "NExT-QA"
    hf_repo: Optional[str] = None   # e.g., "lmms-lab/NExT-QA"
    license: str = "follow_source_dataset"
    is_rl_native: bool = False      # True for Video-R1, SpaceR, NEP

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / self.dataset_name
        self.video_dir = self.raw_dir / "videos"
        self.ann_dir = self.raw_dir / "annotations"

    # ─── Download ─────────────────────────────────────────────────────────

    def generate_download_script(self) -> str:
        """Generate shell commands to download this dataset.

        Uses hf_dl.sh for HuggingFace datasets.
        Returns manual instructions for non-HF datasets.
        """
        if self.hf_repo is None:
            return self._generate_manual_download_instructions()

        config = _load_download_config()
        dl = config["download"]
        hf_dl = dl["hf_dl_script"]
        browser = dl["browser_executable_path"]
        concurrency = dl["scrape_concurrency"]

        lines = [
            f"# === {self.display_name} ({self.dataset_name}) ===",
            f"echo 'Downloading {self.display_name} from {self.hf_repo}...'",
            f'bash {hf_dl} {self.hf_repo} dataset \\',
            f'    --browser-executable-path "{browser}" \\',
            f'    --scrape-concurrency {concurrency}',
            f"",
            f"# Post-download: organize into expected structure",
            f"mkdir -p {self.raw_dir}/videos {self.raw_dir}/annotations",
            self._post_download_commands(),
            "",
        ]
        return "\n".join(lines)

    def _generate_manual_download_instructions(self) -> str:
        """For non-HF datasets, generate a comment block with instructions."""
        return (
            f"# === {self.display_name} ({self.dataset_name}) ===\n"
            f"# MANUAL DOWNLOAD REQUIRED\n"
            f"# This dataset is not hosted on HuggingFace.\n"
            f"# Please download manually and place files in:\n"
            f"#   {self.raw_dir}/videos/\n"
            f"#   {self.raw_dir}/annotations/\n"
            f"# See configs/download_config.yaml for source URL.\n"
            f"mkdir -p {self.raw_dir}/videos {self.raw_dir}/annotations\n"
        )

    def _post_download_commands(self) -> str:
        """Override to add dataset-specific post-download commands."""
        return "# (no post-download steps needed)"

    # ─── Data iteration ───────────────────────────────────────────────────

    @abstractmethod
    def iterate_raw(self, split: str = "train") -> Iterator[dict]:
        """Yield raw annotation dicts from the specified split.

        Each dict should contain at minimum:
        - video_id: str
        - question: str
        - answer: str
        - Any other dataset-specific fields
        """
        ...

    @abstractmethod
    def to_canonical(
        self,
        raw: dict,
        sub_capability: SubCapability,
        task_type: TaskType,
    ) -> CanonicalSample:
        """Convert one raw annotation to the canonical schema."""
        ...

    # ─── Helpers ──────────────────────────────────────────────────────────

    def video_path(self, video_id: str, ext: str = "mp4") -> str:
        """Return file:// URI for a video."""
        abs_path = (self.video_dir / f"{video_id}.{ext}").resolve()
        return f"file://{abs_path}"

    def make_data_id(self, raw_id: str) -> str:
        """Create a globally unique data_id."""
        return f"{self.dataset_name}_{raw_id}"

    def build_type(self) -> BuildType:
        if self.is_rl_native:
            return BuildType.NATIVE
        return BuildType.CONVERTED

    def make_messages(
        self,
        video_url: str,
        question_text: str,
        answer_text: str,
        system_prompt: str = "You are a video reasoning assistant. Follow the required answer format exactly.",
        subtitle_text: Optional[str] = None,
        clip_fps: Optional[float] = None,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
    ) -> list[Message]:
        """Build the standard messages list."""
        user_content: list = [
            VideoURLContent(
                video_url=VideoURL(
                    url=video_url,
                    clip_fps=clip_fps,
                    start_s=start_s,
                    end_s=end_s,
                )
            ),
            TextContent(text=question_text),
        ]
        if subtitle_text:
            user_content.append(TextContent(text=f"Optional subtitle/context: {subtitle_text}"))

        return [
            Message(role="system", content=[TextContent(text=system_prompt)]),
            Message(role="user", content=user_content),
            Message(role="assistant", content=[TextContent(text=answer_text)]),
        ]

    def make_mcq_grader(self, gt: str) -> Grader:
        """Create a standard MCQ grader."""
        return Grader(
            type="ruler",
            name="option_match",
            gt=gt,
            params=GraderParams(min_score=1.0),
        )

    def make_exact_match_grader(self, gt: str) -> Grader:
        return Grader(
            type="ruler",
            name="exact_match",
            gt=gt,
            params=GraderParams(min_score=1.0),
        )
