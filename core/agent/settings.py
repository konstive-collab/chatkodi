"""Model and assistant settings chosen in the window (runtime/assistant_settings.json).

Answers from documents stay greedy (temperature 0) whatever is set here:
the «creativity» slider affects letters, free writing and small talk only.
Context size, device and vision are applied to llama-server at start.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields

from core.paths import runtime_root

LENGTH_FACTORS = {"short": 0.6, "normal": 1.0, "long": 1.6}


@dataclass
class AssistantSettings:
    creativity: float = 0.3          # letters, free writing, chat
    answer_length: str = "normal"    # short | normal | long
    context_tokens: int = 16384      # 8192 | 16384 | 32768 (restart)
    device: str = "auto"             # auto | gpu | cpu (restart)
    vision: bool = True              # photos and scans (restart)
    voice: str = "irina"             # irina | dmitri
    speech: bool = True              # microphone input
    commands: bool = True            # console commands with confirmation
    suggestions: bool = True         # next-step hints under answers
    animate: bool = True             # answers appear word by word
    autonomy: str = "ask"            # ask | auto | full (see core.agent.permissions)
    memory: bool = True              # remember stable facts between chats
    search_folders: list = None      # extra folders to look for files (besides the library and user folders)

    def __post_init__(self):
        if self.search_folders is None:
            self.search_folders = []

    @property
    def length_factor(self) -> float:
        return LENGTH_FACTORS.get(self.answer_length, 1.0)


def settings_path():
    return runtime_root() / "assistant_settings.json"


def load_settings() -> AssistantSettings:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        known = {f.name for f in fields(AssistantSettings)}
        return AssistantSettings(**{k: v for k, v in data.items() if k in known})
    except Exception:
        return AssistantSettings()


def save_settings(settings: AssistantSettings) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")


def apply_to_environment(settings: AssistantSettings) -> None:
    """Values read by llama-server start-up (set before the model loads).

    Only settings the user saved in the window are applied; variables set by an
    administrator in the environment are never removed."""
    if not settings_path().is_file():
        return
    os.environ["CHATKODI_LLM_CTX"] = str(int(settings.context_tokens))
    if settings.device == "cpu":
        os.environ["CHATKODI_FORCE_CPU"] = "1"
    if not settings.vision:
        os.environ["CHATKODI_DISABLE_VISION"] = "1"
