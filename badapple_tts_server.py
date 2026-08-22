#!/Users/savag3/bad_apple/.venv/bin/python
"""Bad Apple local neural TTS server.

Runs Piper TTS on a Unix socket. The menu bar sends text, gets back a path to a
WAV file, and plays it. No cloud, no API, no subscription.

Socket protocol (line-delimited JSON):

  request : {"text": "Babe, like... listen.", "voice": "en_US-lessac-high"}
  response: {"ok": true, "wav_path": "/tmp/badapple_tts_xxx.wav", "sample_rate": 22050,
             "duration_ms": 1420}
  error   : {"ok": false, "error": "voice model not found"}

Environment:
  BADAPPLE_TTS_VOICE       default voice name (default: es_MX-claude-high)
  BADAPPLE_TTS_VOICES_DIR  directory with .onnx and .onnx.json files
                           (default: ./voices next to this script)
  BADAPPLE_TTS_SOCKET      Unix socket path (default: /tmp/badapple_tts.sock)
  BADAPPLE_TTS_LENGTH_SCALE  voice speed, <1 faster >1 slower (default: 1.05)
  BADAPPLE_TTS_NOISE_SCALE   generator noise, higher = more natural (default: 0.80)
  BADAPPLE_TTS_NOISE_W_SCALE phoneme width noise (default: 0.80)
  BADAPPLE_TTS_VOLUME        output gain (default: 0.95)
"""

import json
import os
import re
import socket
import sys
import tempfile
import threading
import wave
from pathlib import Path

from piper.config import SynthesisConfig
from piper.voice import PiperVoice

DEFAULT_VOICE = os.environ.get("BADAPPLE_TTS_VOICE", "en_US-amy-medium")
DEFAULT_SOCKET = os.environ.get("BADAPPLE_TTS_SOCKET", "/tmp/badapple_tts.sock")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VOICES_DIR = os.environ.get("BADAPPLE_TTS_VOICES_DIR", str(SCRIPT_DIR / "voices"))

# Piper voices are not shipped in git (they are ~60 MB). The server resolves the
# model relative to the voices dir. If missing, it prints a one-time download
# command and returns an error.

def _voices_dir() -> Path:
    return Path(DEFAULT_VOICES_DIR).expanduser().resolve()


def _voice_paths(name: str):
    d = _voices_dir()
    onnx = d / f"{name}.onnx"
    config = d / f"{name}.onnx.json"
    return onnx, config


def _download_url(name: str) -> tuple[str, str]:
    # Convert voice name like es_MX-claude-high or en_US-lessac-high
    # to the HuggingFace path: en/en_US/lessac/high/en_US-lessac-high
    parts = name.split("-")
    if len(parts) < 3:
        raise ValueError(f"voice name '{name}' does not match locale-speaker-quality pattern")
    locale = "-".join(parts[: len(parts) - 2])
    speaker = parts[-2]
    quality = parts[-1]
    lang = locale.split("_")[0]
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{locale}/{speaker}/{quality}/{name}"
    return f"{base}.onnx?download=true", f"{base}.onnx.json?download=true"


def _ensure_voice(name: str):
    onnx, config = _voice_paths(name)
    if onnx.exists() and config.exists():
        return onnx, config

    d = _voices_dir()
    d.mkdir(parents=True, exist_ok=True)
    onnx_url, config_url = _download_url(name)

    print(f"badapple_tts: downloading voice {name}...", file=sys.stderr)
    print(f"  {onnx_url}", file=sys.stderr)
    try:
        import urllib.request
        urllib.request.urlretrieve(onnx_url, str(onnx))
        urllib.request.urlretrieve(config_url, str(config))
    except Exception as e:
        raise RuntimeError(
            f"could not download voice {name}: {e}\n"
            f"Run manually from {d}:\n"
            f"  curl -L -o {onnx.name} '{onnx_url}'\n"
            f"  curl -L -o {config.name} '{config_url}'"
        ) from e

    return onnx, config


_lock = threading.Lock()
_voice_cache: dict[str, PiperVoice] = {}

def _load_voice(name: str) -> PiperVoice:
    with _lock:
        if name not in _voice_cache:
            onnx, config = _ensure_voice(name)
            _voice_cache[name] = PiperVoice.load(str(onnx), config_path=str(config))
        return _voice_cache[name]


def _synth_config() -> SynthesisConfig:
    return SynthesisConfig(
        length_scale=float(os.environ.get("BADAPPLE_TTS_LENGTH_SCALE", "0.95")),
        noise_scale=float(os.environ.get("BADAPPLE_TTS_NOISE_SCALE", "0.80")),
        noise_w_scale=float(os.environ.get("BADAPPLE_TTS_NOISE_W_SCALE", "0.80")),
        volume=float(os.environ.get("BADAPPLE_TTS_VOLUME", "0.95")),
        normalize_audio=True,
    )


def _clean_text(text: str) -> str:
    # Keep only speakable characters; drop formatting and URLs.
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("*", "")
    text = re.sub(r"[ʋʌɑɒɛɪʊɔəæ]", lambda m: {"ʋ":"v","ʌ":"v","ɑ":"a","ɒ":"o","ɛ":"e","ɪ":"i","ʊ":"u","ɔ":"o","ə":"a","æ":"a"}[m.group()], text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    return text.strip()


def _synthesize(text: str, voice_name: str = DEFAULT_VOICE) -> Path:
    text = _clean_text(text)
    if not text:
        raise ValueError("empty text after cleaning")

    voice = _load_voice(voice_name)
    cfg = _synth_config()

    fd, wav_path = tempfile.mkstemp(prefix="badapple_tts_", suffix=".wav", dir="/tmp")
    os.close(fd)

    with wave.open(wav_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=cfg)

    return Path(wav_path)


def _handle_request(raw: bytes) -> dict:
    try:
        req = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"invalid JSON: {e}"}

    text = req.get("text")
    if not text or not isinstance(text, str):
        return {"ok": False, "error": "missing or invalid 'text' field"}

    voice_name = req.get("voice", DEFAULT_VOICE)
    text_preview = text[:60].replace("\n", " ")
    print(f"badapple_tts: request voice={voice_name} text='{text_preview}...'", file=sys.stderr)
    try:
        wav_path = _synthesize(text, voice_name)
        with wave.open(str(wav_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration_ms = int((frames / rate) * 1000) if rate else 0
        print(f"badapple_tts: synthesized {duration_ms}ms -> {wav_path}", file=sys.stderr)
        return {
            "ok": True,
            "wav_path": str(wav_path),
            "sample_rate": rate,
            "duration_ms": duration_ms,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _send_json(conn: socket.socket, obj: dict):
    payload = (json.dumps(obj) + "\n").encode("utf-8")
    conn.sendall(payload)


def _serve_client(conn: socket.socket):
    chunks = []
    conn.settimeout(None)
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
        raw = b"".join(chunks)
        if raw:
            response = _handle_request(raw)
            _send_json(conn, response)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _warm_voice():
    try:
        _load_voice(DEFAULT_VOICE)
        print(f"badapple_tts: voice {DEFAULT_VOICE} loaded", file=sys.stderr)
    except Exception as e:
        print(f"badapple_tts: failed to preload voice: {e}", file=sys.stderr)


def main():
    socket_path = Path(DEFAULT_SOCKET).expanduser()
    if socket_path.exists():
        try:
            socket_path.unlink()
        except OSError as e:
            print(f"badapple_tts: cannot remove stale socket: {e}", file=sys.stderr)
            sys.exit(1)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(4)
    except OSError as e:
        print(f"badapple_tts: cannot bind {socket_path}: {e}", file=sys.stderr)
        sys.exit(1)

    os.chmod(str(socket_path), 0o666)
    print(f"badapple_tts: listening on {socket_path} (voice={DEFAULT_VOICE}, voices_dir={_voices_dir()})", file=sys.stderr)

    # Preload so first request is fast; if download fails, server still starts
    # and returns errors per request.
    _warm_voice()

    try:
        while True:
            conn, _ = server.accept()
            client = threading.Thread(target=_serve_client, args=(conn,), daemon=True)
            client.start()
    except KeyboardInterrupt:
        print("badapple_tts: shutting down", file=sys.stderr)
    finally:
        try:
            socket_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
