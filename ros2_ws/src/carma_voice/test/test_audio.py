"""WAV clip decoding."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from carma_voice.audio import WHISPER_RATE, decode_wav, encode_wav


def test_round_trip_at_whisper_rate() -> None:
    tone = 0.5 * np.sin(np.linspace(0, 200 * np.pi, WHISPER_RATE)).astype(np.float32)
    samples, duration = decode_wav(encode_wav(tone, WHISPER_RATE))
    assert duration == pytest.approx(1.0)
    assert samples.shape == tone.shape
    assert np.allclose(samples, tone, atol=1e-3)


def test_resamples_48k_to_16k() -> None:
    clip = encode_wav(np.zeros(48_000, np.float32), 48_000)
    samples, duration = decode_wav(clip)
    assert duration == pytest.approx(1.0)
    assert samples.size == WHISPER_RATE


def test_stereo_is_averaged_to_mono() -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(WHISPER_RATE)
        wav.writeframes(np.array([16384, -16384] * 100, dtype="<i2").tobytes())
    samples, _ = decode_wav(buf.getvalue())
    assert samples.size == 100
    assert np.allclose(samples, 0.0)


def test_rejects_non_wav() -> None:
    with pytest.raises(ValueError, match="WAV"):
        decode_wav(b"definitely not audio")
