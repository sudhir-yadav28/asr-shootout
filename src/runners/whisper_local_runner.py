"""OpenAI Whisper-large-v3, run locally via HuggingFace transformers (GPU needed)."""
from pathlib import Path

from src.runners.base import BaseRunner


class WhisperLocalRunner(BaseRunner):
    name = "whisper_large_v3_local"

    def __init__(self, cache_root: Path, language: str = "hi"):
        super().__init__(cache_root)
        # Lazy import — only loaded on Colab where torch/transformers are installed.
        import torch
        from transformers import pipeline as hf_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=dtype,
            device=device,
        )
        self.language = language

    def _transcribe(self, audio_path: Path) -> tuple[str, dict]:
        result = self.pipe(
            str(audio_path),
            generate_kwargs={"language": self.language, "task": "transcribe"},
            return_timestamps=False,
        )
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip(), {"text": text}
