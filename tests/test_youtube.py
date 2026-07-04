import asyncio
import logging

import pytest

from lingus.adapters import youtube


@pytest.mark.asyncio
async def test_resolve_audio_url_uses_yt_dlp_api(monkeypatch):
    calls = []

    async def fake_resolve_media_url(url, fmt, *, label, timeout):
        calls.append((url, fmt, label, timeout))
        return "https://media.example/audio"

    monkeypatch.setattr(youtube, "resolve_media_url", fake_resolve_media_url)
    adapter = youtube.YouTubeCaptureAdapter("video-id", resolve_timeout=12.0)

    assert await adapter._resolve_audio_url() == "https://media.example/audio"
    assert calls == [
        (
            "https://www.youtube.com/watch?v=video-id",
            "bestaudio/best",
            "audio",
            12.0,
        )
    ]


@pytest.mark.asyncio
async def test_resolve_audio_url_wraps_yt_dlp_errors(monkeypatch):
    async def fake_resolve_media_url(*args, **kwargs):
        raise youtube.YtDlpError("yt-dlp timed out")

    monkeypatch.setattr(youtube, "resolve_media_url", fake_resolve_media_url)
    adapter = youtube.YouTubeCaptureAdapter("video-id")

    with pytest.raises(RuntimeError, match="yt-dlp timed out"):
        await adapter._resolve_audio_url()


class TimeoutThenExitProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.waits = 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.waits += 1
        if not self.killed:
            raise TimeoutError
        return self.returncode


@pytest.mark.asyncio
async def test_stop_reaps_ffmpeg_after_kill():
    adapter = youtube.YouTubeCaptureAdapter("video-id")
    proc = TimeoutThenExitProcess()
    adapter._ff = proc

    await adapter.stop()

    assert proc.terminated
    assert proc.killed
    assert proc.waits == 2
    assert adapter._ff is None


class FakeRawVideoStdout:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def readexactly(self, n: int) -> bytes:
        if not self._chunks:
            raise asyncio.IncompleteReadError(partial=b"", expected=n)
        chunk = self._chunks.pop(0)
        if len(chunk) != n:
            raise asyncio.IncompleteReadError(partial=chunk, expected=n)
        return chunk


class FakeVideoProcess:
    def __init__(self, stdout: FakeRawVideoStdout) -> None:
        self.stdout = stdout
        self.stderr = None
        self.returncode = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_video_frames_yields_raw_rgb_frames(monkeypatch):
    adapter = youtube.YouTubeCaptureAdapter(
        "video-id", frame_width=2, frame_height=2, frame_fps=2.0
    )
    frame_size = 2 * 2 * 3
    proc = FakeVideoProcess(
        FakeRawVideoStdout([b"a" * frame_size, b"b" * frame_size])
    )
    calls = []

    async def fake_resolve_video_url():
        return "https://media.example/video"

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        return proc

    monkeypatch.setattr(adapter, "_resolve_video_url", fake_resolve_video_url)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )

    frames = []
    async for frame in adapter.video_frames():
        frames.append(frame)

    assert [f.data for f in frames] == [b"a" * frame_size, b"b" * frame_size]
    assert [(f.width, f.height, f.ts) for f in frames] == [(2, 2, 0.0), (2, 2, 0.5)]
    assert calls
    assert "-pix_fmt" in calls[0]
    assert "rgb24" in calls[0]


class OneChunkStderr:
    def __init__(self, chunk: bytes) -> None:
        self._chunks = [chunk, b""]

    async def read(self, n: int) -> bytes:
        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_ffmpeg_stderr_warns_while_running(caplog):
    adapter = youtube.YouTubeCaptureAdapter("video-id")
    with caplog.at_level(logging.DEBUG, logger="lingus.adapters._ffmpeg"):
        await adapter._drain_ffmpeg_stderr(OneChunkStderr(b"HLS reconnecting"))
    record = next(r for r in caplog.records if "HLS reconnecting" in r.message)
    assert record.levelno == logging.WARNING


@pytest.mark.asyncio
async def test_ffmpeg_stderr_downgraded_during_teardown(caplog):
    # ctrl-c: terminate() makes ffmpeg dump benign exit lines. Those must not
    # surface as warnings once stop() has flipped the teardown flag.
    adapter = youtube.YouTubeCaptureAdapter("video-id")
    adapter._stopping = True
    with caplog.at_level(logging.DEBUG, logger="lingus.adapters._ffmpeg"):
        await adapter._drain_ffmpeg_stderr(OneChunkStderr(b"Immediate exit requested"))
    record = next(r for r in caplog.records if "Immediate exit requested" in r.message)
    assert record.levelno == logging.DEBUG


@pytest.mark.asyncio
async def test_stop_reaps_video_process():
    adapter = youtube.YouTubeCaptureAdapter("video-id")
    proc = FakeVideoProcess(FakeRawVideoStdout([]))
    adapter._vf = proc

    await adapter.stop()

    assert proc.terminated
    assert adapter._vf is None
