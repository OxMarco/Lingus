"""Pre-ASR audio gating.

Whisper can hallucinate transcript over music, especially song lyrics and outro
music. The gate runs before ASR and suppresses windows that look more like music
or silence than streamer speech, while preserving timing by replacing dropped
windows with short silence.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from typing import Any

from ..adapters.base import AudioChunk
from ..logging import get_logger
from .base import AudioGateBackend, AudioGateDecision

log = get_logger(__name__)

_BYTES_PER_SAMPLE = 2
_SPEECH_LABELS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "dialogue",
    "talking",
)
_MUSIC_LABELS = (
    "music",
    "singing",
    "song",
    "rapping",
    "musical instrument",
    "guitar",
    "piano",
    "drum",
    "choir",
)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(v, lo), hi)


class SpectralAudioGate(AudioGateBackend):
    """Dependency-light speech/music gate based on energy and spectrum shape.

    This is intentionally conservative: it drops silence and obvious sustained
    music, but passes unknown/mixed windows so ASR confidence and the existing
    hallucination filter still get a say.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 2.0,
        speech_threshold: float = 0.45,
        music_threshold: float = 0.65,
        silence_rms: float = 0.003,
        replacement_silence_seconds: float = 0.5,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0.0 <= speech_threshold <= 1.0:
            raise ValueError("speech_threshold must be between 0 and 1")
        if not 0.0 <= music_threshold <= 1.0:
            raise ValueError("music_threshold must be between 0 and 1")
        if silence_rms < 0:
            raise ValueError("silence_rms must be non-negative")
        if replacement_silence_seconds < 0:
            raise ValueError("replacement_silence_seconds must be non-negative")
        self.window_seconds = window_seconds
        self.speech_threshold = speech_threshold
        self.music_threshold = music_threshold
        self.silence_rms = silence_rms
        self.replacement_silence_seconds = replacement_silence_seconds

    async def gate_stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
        window: list[AudioChunk] = []
        sample_rate: int | None = None
        samples = 0
        async for chunk in chunks:
            if sample_rate is not None and chunk.sample_rate != sample_rate and window:
                async for out in self._flush(window, sample_rate):
                    yield out
                window = []
                samples = 0
            sample_rate = chunk.sample_rate
            window.append(chunk)
            samples += len(chunk.pcm) // _BYTES_PER_SAMPLE
            if samples >= int(self.window_seconds * chunk.sample_rate):
                async for out in self._flush(window, chunk.sample_rate):
                    yield out
                window = []
                samples = 0
        if window and sample_rate is not None:
            async for out in self._flush(window, sample_rate):
                yield out

    async def _flush(
        self, window: list[AudioChunk], sample_rate: int
    ) -> AsyncIterator[AudioChunk]:
        pcm = b"".join(chunk.pcm for chunk in window)
        decision = await asyncio.to_thread(self.classify, pcm, sample_rate)
        if decision.allow_asr:
            for chunk in window:
                yield chunk
            return
        log.debug(
            "audio gate dropped %s window: speech=%.2f music=%.2f (%s)",
            decision.mode,
            decision.speech_score,
            decision.music_score,
            decision.reason,
        )
        silence = self._replacement_silence(window[0].ts, sample_rate)
        if silence is not None:
            yield silence

    def _replacement_silence(self, ts: float, sample_rate: int) -> AudioChunk | None:
        samples = int(self.replacement_silence_seconds * sample_rate)
        if samples <= 0:
            return None
        return AudioChunk(pcm=b"\0" * samples * _BYTES_PER_SAMPLE, sample_rate=sample_rate, ts=ts)

    def classify(self, pcm: bytes, sample_rate: int) -> AudioGateDecision:
        import numpy as np

        if not pcm:
            return AudioGateDecision(False, "silence", reason="empty window")
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return AudioGateDecision(False, "silence", reason="empty samples")
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < self.silence_rms:
            return AudioGateDecision(False, "silence", reason=f"rms {rms:.4f}")

        frame_len = max(1, int(0.025 * sample_rate))
        hop = max(1, int(0.010 * sample_rate))
        frames = _framed(audio, frame_len, hop)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        active_ratio = float(np.mean(frame_rms > max(self.silence_rms * 1.5, rms * 0.2)))
        rms_cv = float(np.std(frame_rms) / (np.mean(frame_rms) + 1e-6))
        speech_band, centroid_norm = _spectral_shape(audio, sample_rate)

        # Speech tends to have pauses/bursts and a strong 300-3400 Hz band.
        speech_score = _clamp(
            0.25 * speech_band
            + 0.45 * _clamp((rms_cv - 0.25) / 0.85)
            + 0.20 * _clamp((0.98 - active_ratio) / 0.55)
            + 0.10 * _clamp((0.28 - centroid_norm) / 0.28)
        )
        # Music-only beds tend to be sustained and smoother across short frames.
        music_score = _clamp(
            0.40 * active_ratio
            + 0.35 * _clamp((0.65 - rms_cv) / 0.65)
            + 0.15 * _clamp((speech_band - 0.35) / 0.65)
            + 0.10 * _clamp((centroid_norm - 0.05) / 0.30)
        )

        if music_score >= self.music_threshold and speech_score < self.speech_threshold:
            return AudioGateDecision(
                False,
                "music",
                speech_score=speech_score,
                music_score=music_score,
                reason="sustained music-like spectrum",
            )
        if music_score >= self.music_threshold and speech_score >= self.speech_threshold:
            return AudioGateDecision(
                True,
                "mixed",
                speech_score=speech_score,
                music_score=music_score,
                reason="speech-like content over music",
            )
        mode = "speech" if speech_score >= self.speech_threshold else "unknown"
        return AudioGateDecision(
            True,
            mode,
            speech_score=speech_score,
            music_score=music_score,
            reason="conservative pass",
        )


class HFAudioClassifierGate(SpectralAudioGate):
    """Hugging Face AudioSet classifier gate.

    Defaults to MIT's AST AudioSet model. It is heavier than the spectral gate,
    but gives semantic labels such as Speech, Music, Singing, and Rapping.
    """

    def __init__(
        self,
        *,
        model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        top_k: int = 12,
        cache_dir: str | None = ".lingus/hf/hub",
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
                pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hf_ast audio gate needs transformers/torch "
                "(pip install -e '.[audio-ml]')"
            ) from exc
        self.model = model
        self.top_k = top_k
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model_obj = AutoModelForAudioClassification.from_pretrained(
            model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._pipe = pipeline(
            "audio-classification",
            model=model_obj,
            feature_extractor=feature_extractor,
        )

    def classify(self, pcm: bytes, sample_rate: int) -> AudioGateDecision:
        import numpy as np

        if not pcm:
            return AudioGateDecision(False, "silence", reason="empty window")
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if float(np.sqrt(np.mean(audio * audio))) < self.silence_rms:
            return AudioGateDecision(False, "silence", reason="rms below threshold")
        result = self._pipe({"array": audio, "sampling_rate": sample_rate}, top_k=self.top_k)
        labels = _normalize_hf_result(result)
        speech_score = _max_label_score(labels, _SPEECH_LABELS)
        music_score = _max_label_score(labels, _MUSIC_LABELS)
        if music_score >= self.music_threshold and speech_score < self.speech_threshold:
            return AudioGateDecision(
                False,
                "music",
                speech_score=speech_score,
                music_score=music_score,
                reason=_top_label(labels),
            )
        mode = "mixed" if music_score >= self.music_threshold else "speech"
        if speech_score < self.speech_threshold:
            mode = "unknown"
        return AudioGateDecision(
            True,
            mode,
            speech_score=speech_score,
            music_score=music_score,
            reason=_top_label(labels),
        )


def _framed(audio: Any, frame_len: int, hop: int) -> Any:
    import numpy as np

    if audio.size < frame_len:
        padded = np.zeros(frame_len, dtype=np.float32)
        padded[: audio.size] = audio
        return padded.reshape(1, frame_len)
    starts = range(0, audio.size - frame_len + 1, hop)
    return np.stack([audio[i : i + frame_len] for i in starts])


def _spectral_shape(audio: Any, sample_rate: int) -> tuple[float, float]:
    import numpy as np

    n = int(2 ** math.ceil(math.log2(max(256, min(audio.size, sample_rate * 2)))))
    window = audio[:n]
    if window.size < n:
        padded = np.zeros(n, dtype=np.float32)
        padded[: window.size] = window
        window = padded
    spectrum = np.abs(np.fft.rfft(window * np.hanning(n))) + 1e-8
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 0:
        return 0.0, 0.0
    speech_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    speech_band = float(np.sum(spectrum[speech_mask]) / total)
    centroid = float(np.sum(freqs * spectrum) / total)
    centroid_norm = _clamp(centroid / (sample_rate / 2))
    return speech_band, centroid_norm


def _normalize_hf_result(result: Any) -> list[dict[str, float | str]]:
    if result and isinstance(result[0], list):
        result = result[0]
    return [
        {"label": str(item.get("label", "")), "score": float(item.get("score", 0.0))}
        for item in result
        if isinstance(item, dict)
    ]


def _max_label_score(labels: list[dict[str, float | str]], needles: tuple[str, ...]) -> float:
    best = 0.0
    for item in labels:
        label = str(item["label"]).lower()
        if any(needle in label for needle in needles):
            best = max(best, float(item["score"]))
    return best


def _top_label(labels: list[dict[str, float | str]]) -> str:
    if not labels:
        return "no labels"
    item = max(labels, key=lambda row: float(row["score"]))
    return f"{item['label']} {float(item['score']):.2f}"
