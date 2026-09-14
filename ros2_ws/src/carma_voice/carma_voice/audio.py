"""WAV decoding for speech clips. No ROS imports, so it is unit-testable anywhere."""

from __future__ import annotations

import io
import wave

import numpy as np

WHISPER_RATE = 16_000


def decode_wav(data: bytes, target_rate: int = WHISPER_RATE) -> tuple[np.ndarray, float]:
    """Decode 16-bit PCM WAV bytes into mono float32 samples at ``target_rate``.

    Multi-channel audio is averaged to mono; other sample rates are resampled
    linearly, which is adequate for speech recognition input.

    Returns:
        The samples in ``[-1, 1]`` and the clip duration in seconds.

    Raises:
        ValueError: The bytes are not a 16-bit PCM WAV file.
    """
    try:
        with wave.open(io.BytesIO(data)) as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        msg = f"not a readable WAV clip: {exc}"
        raise ValueError(msg) from exc
    if width != 2:
        msg = f"expected 16-bit PCM, got {8 * width}-bit"
        raise ValueError(msg)

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    duration_s = samples.size / float(rate) if rate else 0.0
    if rate != target_rate and samples.size:
        n_out = max(1, round(samples.size * target_rate / rate))
        positions = np.linspace(0.0, samples.size - 1, n_out)
        samples = np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
    return samples, duration_s


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    """Encode mono samples (int16, or float in ``[-1, 1]``) as 16-bit PCM WAV."""
    if samples.dtype != np.int16:
        samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return buf.getvalue()
