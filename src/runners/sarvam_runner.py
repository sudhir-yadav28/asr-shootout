"""Sarvam AI Saarika v2.5 — India-tuned API model, designed for Hindi-English code-switch."""
from pathlib import Path
import requests

from src.config import SARVAM_API_KEY
from src.runners.base import BaseRunner


SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamRunner(BaseRunner):
    name = "sarvam_saarika_v2_5"

    def __init__(self, cache_root: Path, model: str = "saarika:v2.5", language: str = "hi-IN"):
        super().__init__(cache_root)
        if not SARVAM_API_KEY:
            raise RuntimeError("SARVAM_API_KEY missing in .env")
        self.headers = {"api-subscription-key": SARVAM_API_KEY}
        self.model = model
        self.language = language

    def _transcribe(self, audio_path: Path) -> tuple[str, dict]:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {"model": self.model, "language_code": self.language}
            r = requests.post(
                SARVAM_STT_URL, headers=self.headers, files=files, data=data, timeout=60
            )
        r.raise_for_status()
        raw = r.json()
        text = raw.get("transcript", "") or raw.get("text", "")
        return text, raw
