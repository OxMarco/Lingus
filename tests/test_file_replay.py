import pytest

from lingus.adapters import file_replay
from lingus.adapters.file_replay import iter_jsonl, paced_rows


@pytest.mark.asyncio
async def test_paced_rows_respects_row_timestamps(monkeypatch):
    sleeps = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(file_replay.asyncio, "sleep", fake_sleep)
    rows = [{"t": 0.5, "text": "first"}, {"t": 1.5, "text": "second"}]

    seen = [row async for row in paced_rows(rows, speed=10.0)]

    assert seen == rows
    assert sleeps == [0.05, 0.1]


def test_iter_jsonl_reports_file_and_line_for_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSONL"):
        list(iter_jsonl(path))
