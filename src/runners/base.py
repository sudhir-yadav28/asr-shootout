"""Base class for ASR runners. Handles caching + timing + error capture."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
import json
import time
import traceback


def _json_default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, bytes):
        return o.decode("utf-8", errors="replace")
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


@dataclass
class TranscriptResult:
    model: str
    clip_id: str
    filename: str
    text: str
    latency_s: float
    error: str | None = None
    raw: dict | None = None


class BaseRunner(ABC):
    name: str = "base"

    def __init__(self, cache_root: Path):
        self.cache_dir = Path(cache_root) / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, clip_id: str) -> Path:
        return self.cache_dir / f"{clip_id}.json"

    def load_cached(self, clip_id: str) -> TranscriptResult | None:
        p = self._cache_path(clip_id)
        if not p.exists():
            return None
        with p.open() as f:
            data = json.load(f)
        return TranscriptResult(**data)

    def save(self, result: TranscriptResult) -> None:
        with self._cache_path(result.clip_id).open("w") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2,
                      default=_json_default)

    @abstractmethod
    def _transcribe(self, audio_path: Path) -> tuple[str, dict]:
        """Returns (transcript_text, raw_response_dict). Must be implemented per system."""

    def transcribe(self, clip_id: str, filename: str, audio_path: Path,
                   *, force: bool = False) -> TranscriptResult:
        if not force:
            cached = self.load_cached(clip_id)
            if cached is not None:
                return cached
        t0 = time.perf_counter()
        try:
            text, raw = self._transcribe(audio_path)
            res = TranscriptResult(
                model=self.name,
                clip_id=clip_id,
                filename=filename,
                text=text,
                latency_s=time.perf_counter() - t0,
                error=None,
                raw=raw,
            )
        except Exception as e:
            res = TranscriptResult(
                model=self.name,
                clip_id=clip_id,
                filename=filename,
                text="",
                latency_s=time.perf_counter() - t0,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                raw=None,
            )
        self.save(res)
        return res
