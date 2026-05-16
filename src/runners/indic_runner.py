"""AI4Bharat IndicConformer 600M — India-tuned, Conformer architecture (not Whisper).

Run on Colab. Falls back to vasista22/whisper-hindi-large-v2 if Conformer fails to load
(NeMo dependency can be fiddly). The fallback is still 'India-tuned open-source', so the
2x2 grid stays intact.
"""
from pathlib import Path

from src.runners.base import BaseRunner


class IndicConformerRunner(BaseRunner):
    name = "indic_conformer"

    def __init__(self, cache_root: Path, language: str = "hi", decoding: str = "ctc"):
        super().__init__(cache_root)
        import torch
        from transformers import AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.language = language
        self.decoding = decoding

    def _transcribe(self, audio_path: Path) -> tuple[str, dict]:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(self.device)
        with torch.no_grad():
            text = self.model(wav, self.language, self.decoding)
        if isinstance(text, (list, tuple)):
            text = text[0]
        return str(text).strip(), {"decoding": self.decoding, "lang": self.language}


class IndicWhisperFallbackRunner(BaseRunner):
    """Fallback if IndicConformer dependencies break on Colab."""
    name = "indic_whisper_fallback"

    def __init__(self, cache_root: Path, model_id: str = "vasista22/whisper-hindi-large-v2"):
        super().__init__(cache_root)
        import torch
        from transformers import pipeline as hf_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=device,
        )

    def _transcribe(self, audio_path: Path) -> tuple[str, dict]:
        result = self.pipe(str(audio_path), return_timestamps=False)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip(), {"text": text}
