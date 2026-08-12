"""Shared output-directory configuration for the local video app."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_ROOT = (
    Path(r"E:\ai口播输出") if os.name == "nt" else PROJECT_ROOT / "outputs"
)
OUTPUTS_ROOT = Path(
    os.environ.get("AI_TALKING_VIDEO_OUTPUT_DIR", str(DEFAULT_OUTPUTS_ROOT))
).expanduser().resolve()
WORK_ROOT = OUTPUTS_ROOT / ".work"
