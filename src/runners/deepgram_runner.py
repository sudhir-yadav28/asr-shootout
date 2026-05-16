"""Deepgram Nova-3 — required baseline. Multilingual ('multi') for Hindi-English code-switch.

Uses deepgram-sdk >=7 client.listen.v1.media.transcribe_file interface.
"""
from pathlib import Path
from deepgram import DeepgramClient

from src.config import DEEPGRAM_API_KEY
from src.runners.base import BaseRunner


class DeepgramRunner(BaseRunner):
    name = "deepgram_nova3"

    def __init__(self, cache_root: Path, model: str = "nova-3", language: str = "multi"):
        super().__init__(cache_root)
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY missing in .env")
        self.client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        self.model = model
        self.language = language

    def _transcribe(self, audio_path: Path) -> tuple[str, dict]:
        with open(audio_path, "rb") as f:
            buffer = f.read()
        response = self.client.listen.v1.media.transcribe_file(
            request=buffer,
            model=self.model,
            language=self.language,
            smart_format=True,
            punctuate=True,
        )
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw = response.to_dict()
        else:
            raw = dict(response) if not isinstance(response, dict) else response
        text = (
            raw.get("results", {})
               .get("channels", [{}])[0]
               .get("alternatives", [{}])[0]
               .get("transcript", "")
        )
        return text, raw
