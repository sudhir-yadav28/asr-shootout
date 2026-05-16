"""Audio prep: ensure every clip is *real* 16 kHz mono PCM WAV before going to ASR.

Phone recorders often save files as `.wav` but with M4A/AAC contents inside. Web ASR
APIs sniff the format and handle it, but local models (Whisper / IndicConformer on
Colab) need genuine RIFF PCM WAV. We detect the mismatch and re-encode via ffmpeg.
"""
from pathlib import Path
import subprocess
import shutil
import wave


SUPPORTED = {".wav", ".m4a", ".mp3", ".ogg", ".flac", ".webm", ".aac"}


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    return path


def is_pcm_wav_16k_mono(path: Path) -> bool:
    """True iff `path` is a genuine PCM RIFF WAV at 16 kHz mono."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"RIFF":
                return False
        with wave.open(str(path), "rb") as w:
            return (w.getframerate() == 16000
                    and w.getnchannels() == 1
                    and w.getsampwidth() == 2)
    except Exception:
        return False


def to_wav_16k_mono(src: Path, dst_dir: Path) -> Path:
    """Return a real 16 kHz mono PCM WAV at dst_dir/<stem>.wav. Idempotent."""
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (src.stem + ".wav")

    # Fast path: dst already exists, is a real wav, and is newer than src.
    if (dst.exists()
            and dst.stat().st_mtime >= src.stat().st_mtime
            and is_pcm_wav_16k_mono(dst)):
        return dst

    ensure_ffmpeg()
    # Re-encode through a temp file so we can overwrite src safely if dst == src.
    tmp = dst_dir / (src.stem + ".__convert__.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-sample_fmt", "s16",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)
    return dst


def prep_all(audio_dir: Path) -> list[Path]:
    """Ensure every audio file in audio_dir is a real 16 kHz mono PCM WAV. Returns paths."""
    audio_dir = Path(audio_dir)
    wavs = []
    for f in sorted(audio_dir.iterdir()):
        if f.suffix.lower() not in SUPPORTED:
            continue
        wav = to_wav_16k_mono(f, audio_dir)
        wavs.append(wav)
    return wavs
