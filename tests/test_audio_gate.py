import math

import pytest

from lingus.adapters.base import AudioChunk
from lingus.models.audio_gate import SpectralAudioGate


def _pcm(samples, amp: float = 0.8) -> bytes:
    import numpy as np

    clipped = np.clip(samples * amp, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def _sine(seconds: float, *, freq: float = 440.0, sr: int = 16_000) -> bytes:
    import numpy as np

    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return _pcm(np.sin(2 * math.pi * freq * t), amp=0.5)


def _speech_like(seconds: float, *, sr: int = 16_000) -> bytes:
    import numpy as np

    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    carrier = np.sin(2 * math.pi * 190 * t) + 0.35 * np.sin(2 * math.pi * 760 * t)
    # Bursty envelope: speech has syllabic on/off structure rather than a fully
    # sustained bed.
    envelope = ((np.sin(2 * math.pi * 4.0 * t) + 1.0) / 2.0) ** 2
    envelope[envelope < 0.18] = 0.0
    return _pcm(carrier * envelope, amp=0.45)


async def _collect(chunks):
    return [chunk async for chunk in chunks]


async def _source(chunks):
    for chunk in chunks:
        yield chunk


def test_spectral_audio_gate_drops_silence():
    gate = SpectralAudioGate()
    decision = gate.classify(b"\0" * 16_000, 16_000)

    assert not decision.allow_asr
    assert decision.mode == "silence"


def test_spectral_audio_gate_drops_sustained_music_like_tone():
    gate = SpectralAudioGate()
    decision = gate.classify(_sine(2.0), 16_000)

    assert not decision.allow_asr
    assert decision.mode == "music"
    assert decision.music_score >= decision.speech_score


def test_spectral_audio_gate_allows_bursty_speech_like_audio():
    gate = SpectralAudioGate()
    decision = gate.classify(_speech_like(2.0), 16_000)

    assert decision.allow_asr
    assert decision.mode in {"speech", "mixed", "unknown"}


@pytest.mark.asyncio
async def test_gate_stream_replaces_dropped_music_with_short_silence():
    gate = SpectralAudioGate(window_seconds=1.0, replacement_silence_seconds=0.25)
    chunk = AudioChunk(pcm=_sine(1.0), sample_rate=16_000, ts=12.0)

    out = await _collect(gate.gate_stream(_source([chunk])))

    assert len(out) == 1
    assert out[0].ts == 12.0
    assert out[0].pcm == b"\0" * int(0.25 * 16_000) * 2
