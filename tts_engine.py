"""TTS engine for the Reporter Hasan (KIKI Retrain (2550)) Bengali voice.

Importable without Streamlit: ``python -c "import tts_engine"`` must work
in a plain venv that only has sherpa-onnx installed.

Model source: public Hugging Face repo ``demoqqq/banglatts-kiki-retrain-2550``.
The 114 MB ONNX exceeds GitHub's 100 MB per-file limit, so the model is
never committed -- it is downloaded at runtime and verified by exact byte
size AND MD5 before use.

Proven synthesis settings (same as the Android app / VM tests):
noise_scale 0.667, noise_scale_w 0.8, length_scale 1.0,
max_num_sentences 1, silence_scale 0.2, sample rate 22050 Hz.
"""

import hashlib
import io
import os
import re
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

# Sample texts (shared by the UI and the local test).
DEFAULT_TEXT = (
    "ঢাকায় আজ সকাল থেকে হালকা বৃষ্টি হচ্ছে। "
    "আবহাওয়া অধিদপ্তর জানিয়েছে, আগামীকালও বৃষ্টি হতে পারে। "
    "রাজধানীর বিভিন্ন এলাকায় যানজট কিছুটা বেড়েছে।"
)

SAMPLE_NEWS = (
    "মিরপুরে আজ সকালে পুলিশের একটি দল অভিযান চালিয়ে চোরাই মালসহ "
    "তিনজনকে আটক করেছে। পুলিশ জানিয়েছে, আটক ব্যক্তিদের বিরুদ্ধে "
    "আগে থেকেই একাধিক মামলা ছিল। এলাকাবাসী অভিযানকে স্বাগত জানিয়েছেন।"
)

SAMPLE_GENERAL = (
    "আসসালামু আলাইকুম, সবাই কেমন আছেন? "
    "আজকের ভিডিওতে আমরা কথা বলবো বাংলাদেশের আবহাওয়া নিয়ে। "
    "ভিডিওটি ভালো লাগলে লাইক করবেন আর চ্যানেলটি সাবস্ক্রাইব করবেন।"
)


# ---------------------------------------------------------------------------
# Model files (KIKI Retrain (2550) — ২,৫৫০টি ক্লিপ, ৯৭ep থেকে continue, ১২ epoch).
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://huggingface.co/demoqqq/banglatts-kiki-retrain-2550"
    "/resolve/main/model_fullmeta.onnx"
)
MODEL_SIZE = 114192736
MODEL_MD5 = "31dd6b9881d1b26759aa77bab73a5cb0"

TOKENS_URL = (
    "https://huggingface.co/demoqqq/banglatts-kiki-retrain-2550"
    "/resolve/main/tokens_sherpa.txt"
)
TOKENS_SIZE = 603
TOKENS_MD5 = "c174dbb8c0e87cb55cffb1f1d5233c02"

SAMPLE_RATE = 22050
MAX_PART_CHARS = 700          # never generate more than this in one call
JOIN_SILENCE_S = 0.5          # silence between parts
NUM_THREADS = int(os.environ.get("BANGALTTS_THREADS", "2"))

# Orthography: the training transcripts use ONLY the decomposed forms
# (U+09AF U+09BC etc.). Typed Bengali usually comes out precomposed, and
# those code points are not in the 92-token vocab, so normalize first.
# Python NFC does NOT compose these -- explicit replacement is required.
ORTHO_NORMALIZE = {
    "\u09df": "\u09af\u09bc",  # য় -> য + ়
    "\u09dc": "\u09a1\u09bc",  # ড় -> ড + ়
    "\u09dd": "\u09a2\u09bc",  # ঢ় -> ঢ + ়
}


def normalize_text(text: str) -> str:
    """Map precomposed য়/ড়/ঢ় to the decomposed forms the model knows."""
    for src, dst in ORTHO_NORMALIZE.items():
        text = text.replace(src, dst)
    return text


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_size: int, expected_md5: str) -> Path:
    """Download a file, verifying exact size AND MD5. Loud failure on mismatch."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "banglatts-streamlit/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    size = tmp.stat().st_size
    if size != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"download size mismatch for {dest.name}: got {size}, "
            f"expected {expected_size}"
        )
    md5 = _md5_of(tmp)
    if md5 != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"download MD5 mismatch for {dest.name}: got {md5}, "
            f"expected {expected_md5}"
        )
    tmp.replace(dest)
    return dest


def default_cache_dir() -> Path:
    override = os.environ.get("BANGALTTS_MODEL_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "model_cache"


def ensure_model(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Return (model_path, tokens_path), downloading + verifying if needed."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    model_path = cache_dir / "model_fullmeta.onnx"
    tokens_path = cache_dir / "tokens_sherpa.txt"
    if not (model_path.exists() and model_path.stat().st_size == MODEL_SIZE
            and _md5_of(model_path) == MODEL_MD5):
        _download(MODEL_URL, model_path, MODEL_SIZE, MODEL_MD5)
    if not (tokens_path.exists() and tokens_path.stat().st_size == TOKENS_SIZE
            and _md5_of(tokens_path) == TOKENS_MD5):
        _download(TOKENS_URL, tokens_path, TOKENS_SIZE, TOKENS_MD5)
    return model_path, tokens_path


# ---------------------------------------------------------------------------
# Chunking: split ONLY at sentence boundaries, never mid-word / mid-phrase.
# Primary boundary is the danda (।). A part is at most 700 chars.
# ---------------------------------------------------------------------------
_DANDA = "।"
_FALLBACK_BOUNDARIES = ("?", "!", ",", "…")


def split_into_parts(text: str, max_chars: int = MAX_PART_CHARS) -> list[str]:
    """Split text into parts of <= max_chars, cutting only at punctuation.

    Order of boundaries tried: danda, then ? ! , …, then word boundaries
    (spaces) as a last resort for an over-long stretch with no punctuation.
    """
    text = " ".join(normalize_text(text).split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # 1. Split into sentences, keeping the danda attached to its sentence.
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch == _DANDA:
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    # 2. Greedily pack sentences into parts. An over-long single sentence
    #    is broken down by the fallback splitter below.
    parts: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            if parts and len(parts[-1]) + 1 + len(sent) <= max_chars:
                parts[-1] = parts[-1] + " " + sent
            else:
                parts.append(sent)
        else:
            parts.extend(_split_long_sentence(sent, max_chars))
    # 3. Safety net: no part may exceed max_chars.
    final: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            final.append(p)
        else:
            final.extend(_split_long_sentence(p, max_chars))
    return [p for p in final if p]


def _split_long_sentence(sent: str, max_chars: int) -> list[str]:
    """Split one over-long sentence at fallback punctuation, then spaces."""
    # Try cutting after ? ! , … first.
    pieces: list[str] = []
    buf = ""
    for ch in sent:
        buf += ch
        if ch in _FALLBACK_BOUNDARIES:
            pieces.append(buf.strip())
            buf = ""
    if buf.strip():
        pieces.append(buf.strip())
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            # Last resort: word boundaries only, never mid-word.
            words = piece.split(" ")
            cur = ""
            for w in words:
                if len(cur) + len(w) + 1 <= max_chars:
                    cur = (cur + " " + w).strip()
                else:
                    if cur:
                        out.append(cur)
                    cur = w
            if cur:
                out.append(cur)
    # Re-pack small pieces so we do not create needless tiny parts.
    packed: list[str] = []
    for p in out:
        if packed and len(packed[-1]) + 1 + len(p) <= max_chars:
            packed[-1] = packed[-1] + " " + p
        else:
            packed.append(p)
    return packed


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
_tts = None


def _load_tts():
    global _tts
    if _tts is not None:
        return _tts
    import sherpa_onnx

    model_path, tokens_path = ensure_model()
    _tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                ),
                num_threads=NUM_THREADS,
            ),
            rule_fsts="",
            max_num_sentences=1,
            silence_scale=0.2,
        )
    )
    return _tts


def synthesize(text: str) -> tuple[np.ndarray, dict]:
    """Synthesize Bengali text -> (float32 mono samples @22050, info dict).

    Chunks by danda (<=700 chars/part) and joins parts with 0.5 s silence.
    """
    parts = split_into_parts(text)
    if not parts:
        raise ValueError("খালি লেখা — কিছু লিখে তারপর বানান।")
    tts = _load_tts()
    samples = np.zeros(0, dtype=np.float32)
    join = np.zeros(int(JOIN_SILENCE_S * SAMPLE_RATE), dtype=np.float32)
    t0 = time.time()
    for part in parts:
        audio = tts.generate(part, sid=0, speed=1.0)
        s = np.asarray(audio.samples, dtype=np.float32)
        samples = np.concatenate([samples, s, join])
    if len(parts) > 1:
        samples = samples[: len(samples) - len(join)]
    wall = time.time() - t0
    duration = len(samples) / SAMPLE_RATE
    info = {
        "parts": len(parts),
        "part_chars": [len(p) for p in parts],
        "duration_s": round(duration, 2),
        "synth_time_s": round(wall, 1),
        "rtf": round(wall / duration, 2) if duration > 0 else None,
        "peak": round(float(np.abs(samples).max()), 3) if len(samples) else 0.0,
        "rms": round(float(np.sqrt((samples ** 2).mean())), 4) if len(samples) else 0.0,
    }
    return samples, info


def samples_to_wav_bytes(samples: np.ndarray) -> bytes:
    """Float32 mono @22050 -> 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def write_wav(path: Path, samples: np.ndarray) -> Path:
    path = Path(path)
    path.write_bytes(samples_to_wav_bytes(samples))
    return path
