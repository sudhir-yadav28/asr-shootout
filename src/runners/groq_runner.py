"""Groq hosted Whisper-large-v3 — same model as local Whisper, but fast (100x+ real-time)."""
from pathlib import Path
from groq import Groq

from src.config import GROQ_API_KEY
from src.runners.base import BaseRunner


class GroqWhisperRunner(BaseRunner):
    name = "groq_whisper_large_v3"

    def __init__(self, cache_root: Path, model: str = "whisper-large-v3", language: str = "hi"):
        super().__init__(cache_root)
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing in .env")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model
        self.language = language

    def _transcribe(self, audio_path: Path) -> tuple[str, dict]:
        with open(audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                file=(audio_path.name, f.read()),
                model=self.model,
                language=self.language,
                response_format="verbose_json",
                temperature=0.0,
            )
        raw = transcription.model_dump() if hasattr(transcription, "model_dump") else dict(transcription)
        text = raw.get("text", "") if isinstance(raw, dict) else getattr(transcription, "text", "")
        return text, raw
