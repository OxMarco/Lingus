"""faster-whisper ASR backend with VAD-based utterance segmentation.

Whisper is not a streaming model, so we wrap it. Instead of chopping audio into
arbitrary fixed windows (which slices sentences mid-word — see the chopped
"fare in futuro per diciamo..." we got in live testing), we segment on *speech
boundaries*: accumulate audio, run Silero VAD over the rolling buffer, and flush
to Whisper at a pause.

The catch: Whisper pads every call to a 30s mel, so each transcription has a
fixed CPU cost — real-time break-even is several seconds of audio per call. Pure
pause-segmentation would flush 1-2s utterances and fall behind live. So we
*amortize*: we only flush on a pause once ~`min_flush_seconds` of audio has
accrued, batching short utterances into one call cut at a clean pause.
A long lull flushes early (don't strand a final short sentence), and a
`window_seconds` cap force-flushes a pauseless monologue so latency stays bounded.

Silero VAD ships inside faster-whisper, so there's no extra dependency. Both the
VAD scan and the transcription run off the event loop (`asyncio.to_thread`;
ctranslate2 and onnxruntime release the GIL).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from ..adapters.base import AudioChunk
from ..logging import get_logger
from .base import ASRBackend, Transcript

log = get_logger(__name__)

_BYTES_PER_SAMPLE = 2  # int16 PCM

# Phrases Whisper hallucinates on music/near-silence — often *confidently*, so a
# logprob/no-speech check alone won't catch them. The subtitle-credit family is
# the usual offender across languages; a few stock "outro" lines round it out.
_HALLUCINATION_MARKERS = (
    "sottotitoli",  # "Sottotitoli a cura di QTSS" (Italian, seen live)
    "qtss",
    "amara.org",
    "sous-titres",  # French
    "subtítulos",  # Spanish
    "untertitel",  # German
    "字幕",  # CJK
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
)


def _is_hallucination(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    """True if a Whisper segment looks like a phantom (drop it, don't react)."""
    if no_speech_prob >= 0.6 or avg_logprob <= -1.0:
        return True
    low = text.lower()
    return any(marker in low for marker in _HALLUCINATION_MARKERS)


class FasterWhisperASR(ASRBackend):
    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "auto",
        language: str | None = None,
        *,
        window_seconds: float = 10.0,
        min_silence_ms: int = 600,
        min_flush_ms: int = 4000,
        lull_flush_ms: int = 2500,
        speech_pad_ms: int = 200,
        min_speech_ms: int = 250,
        sample_rate: int = 16000,
    ) -> None:
        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        # ctranslate2 has no Metal backend; on Apple Silicon ASR runs on CPU.
        # int8 + a ~10s VAD cap gives turbo a fair shot on a high-end local CPU.
        resolved_device = "cpu" if device in ("auto", "metal") else device
        log.info("loading faster-whisper '%s' on %s (int8)…", model_size, resolved_device)
        self._model = WhisperModel(model_size, device=resolved_device, compute_type="int8")
        self._language = language

        self._get_speech_timestamps = get_speech_timestamps
        self._vad_options = VadOptions(
            min_silence_duration_ms=min_silence_ms,  # gap that ends an utterance
            speech_pad_ms=speech_pad_ms,  # keep a little audio around each segment
            min_speech_duration_ms=min_speech_ms,  # ignore lip-smacks/clicks
        )
        self._sr = sample_rate
        # Force-flush cap: a monologue with no pause still gets transcribed.
        self._max_samples = int(window_seconds * sample_rate)
        self._min_silence_samples = int(min_silence_ms / 1000 * sample_rate)
        # Amortization floor: don't flush a pause until ~this much audio has
        # accrued, so each Whisper call clears the ~4s real-time break-even.
        self._min_flush_samples = int(min_flush_ms / 1000 * sample_rate)
        # A pause this long flushes even a short utterance (real lull, not a beat).
        self._lull_samples = int(lull_flush_ms / 1000 * sample_rate)
        # Don't bother running VAD until there's enough audio to contain a pause.
        self._min_analyze_samples = int(0.5 * sample_rate)

        # Warm the model so the first real utterance doesn't eat the ~3.5s
        # cold-start (thread-pool + allocation on the first inference).
        self._transcribe_blocking(np.zeros(sample_rate, dtype=np.int16).tobytes())
        log.info("faster-whisper ready (VAD-segmented, ~%.0fs cap)", window_seconds)

    async def transcribe_stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        buf = bytearray()
        start_ts = 0.0
        async for chunk in chunks:
            if not buf:
                start_ts = chunk.ts
            buf.extend(chunk.pcm)
            # One chunk can complete several utterances (or a flush can leave a
            # second one behind), so drain until the segmenter says "wait".
            while len(buf) >= self._min_analyze_samples * _BYTES_PER_SAMPLE:
                action, n = await asyncio.to_thread(self._segment, bytes(buf))
                if action == "wait":
                    break
                if action == "flush":
                    cut = n * _BYTES_PER_SAMPLE
                    text = await asyncio.to_thread(self._transcribe_blocking, bytes(buf[:cut]))
                    if text:
                        yield Transcript(text=text, ts=start_ts)
                    del buf[:cut]
                    start_ts += n / self._sr
                elif action == "drop":  # leading silence; advance past it
                    del buf[: n * _BYTES_PER_SAMPLE]
                    start_ts += n / self._sr
                if not buf:
                    break
        # End of stream: flush a final utterance, but only if it holds speech
        # (otherwise we'd pay a Whisper call to transcribe trailing silence).
        if buf and self._has_speech(bytes(buf)):
            text = await asyncio.to_thread(self._transcribe_blocking, bytes(buf))
            if text:
                yield Transcript(text=text, ts=start_ts)

    def _segment(self, pcm: bytes) -> tuple[str, int]:
        """Decide what to do with the current buffer. Returns (action, samples).

        - ("flush", n): transcribe the first n samples as one (amortized) utterance.
        - ("drop", n): discard n leading samples of silence.
        - ("wait", 0): keep accumulating.
        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        total = len(audio)
        segs = self._get_speech_timestamps(audio, self._vad_options, sampling_rate=self._sr)
        if not segs:
            # Pure silence/music: keep only a short tail (speech may be starting)
            # so the buffer doesn't grow without bound during quiet stretches.
            if total > self._min_silence_samples:
                return ("drop", total - self._min_silence_samples)
            return ("wait", 0)
        last_end = segs[-1]["end"]  # already padded by speech_pad_ms
        trailing = total - last_end
        if trailing >= self._min_silence_samples:
            # Speaker paused. Flush only if enough audio has accrued to clear the
            # fixed Whisper cost, or the pause is long enough to be a real lull.
            if last_end >= self._min_flush_samples or trailing >= self._lull_samples:
                return ("flush", min(last_end, total))
        if total >= self._max_samples:
            # Still talking past the cap: flush so latency stays bounded.
            return ("flush", total)
        return ("wait", 0)

    def _has_speech(self, pcm: bytes) -> bool:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return bool(self._get_speech_timestamps(audio, self._vad_options, sampling_rate=self._sr))

    def _transcribe_blocking(self, pcm: bytes) -> str:
        # int16 PCM -> float32 in [-1, 1], which is what faster-whisper expects.
        # No vad_filter here: the segmenter already isolated the speech.
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        # Drop hallucinated segments (the "Sottotitoli a cura di…" family Whisper
        # emits on music/near-silence). Keeps the bot from reacting to phantom
        # speech (CLAUDE.md §9).
        kept = [
            seg.text.strip()
            for seg in segments
            if seg.text.strip()
            and not _is_hallucination(seg.text, seg.no_speech_prob, seg.avg_logprob)
        ]
        return " ".join(kept).strip()
