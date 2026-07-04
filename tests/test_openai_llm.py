"""The transport must survive endpoints that reject `max_tokens` and demand
`max_completion_tokens` (newer OpenAI models), swapping the param and caching
the choice."""

from __future__ import annotations

import pytest

from lingus.models.base import ChatTurn
from lingus.models.openai_llm import (
    OpenAICompatLLM,
    _is_unsupported_temperature,
    _is_unsupported_token_param,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _Unsupported400(Exception):
    status_code = 400
    message = (
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )


class _BadTemperature400(Exception):
    status_code = 400
    message = (
        "Unsupported value: 'temperature' does not support 0.9 with this model. "
        "Only the default (1) value is supported."
    )


class _FakeCompletions:
    def __init__(self, reject: set[str] | None) -> None:
        # `reject` names the kwargs that trigger a 400 (e.g. {"max_tokens",
        # "temperature"}); the matching param is rejected until removed/renamed.
        self._reject = reject or set()
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "max_tokens" in self._reject and "max_tokens" in kwargs:
            raise _Unsupported400()
        if "temperature" in self._reject and "temperature" in kwargs:
            raise _BadTemperature400()
        return _FakeResponse("hi")


def _make_llm(reject: set[str] | None) -> tuple[OpenAICompatLLM, _FakeCompletions]:
    llm = OpenAICompatLLM.__new__(OpenAICompatLLM)
    llm._structured_client = None
    llm._model = "gpt-5.4-mini"
    llm._temperature = 0.9
    llm._max_tokens = 120
    llm._token_param = "max_tokens"
    llm._send_temperature = True
    completions = _FakeCompletions(reject)

    class _Chat:
        def __init__(self, c):
            self.completions = c

    class _Client:
        def __init__(self, c):
            self.chat = _Chat(c)

    llm._client = _Client(completions)
    return llm, completions


def test_detects_unsupported_param():
    assert _is_unsupported_token_param(_Unsupported400(), "max_tokens")
    assert not _is_unsupported_token_param(ValueError("nope"), "max_tokens")
    assert not _is_unsupported_token_param(_BadTemperature400(), "max_tokens")


def test_detects_unsupported_temperature():
    assert _is_unsupported_temperature(_BadTemperature400())
    assert not _is_unsupported_temperature(_Unsupported400())
    assert not _is_unsupported_temperature(ValueError("nope"))


@pytest.mark.asyncio
async def test_swaps_to_max_completion_tokens_and_caches():
    llm, completions = _make_llm(reject={"max_tokens"})
    out = await llm.generate([ChatTurn("user", "yo")])
    assert out == "hi"
    # First attempt used max_tokens (rejected), retry used max_completion_tokens.
    assert "max_tokens" in completions.calls[0]
    assert "max_completion_tokens" in completions.calls[1]
    assert llm._token_param == "max_completion_tokens"

    # Second call goes straight to the cached param — no wasted round-trip.
    completions.calls.clear()
    await llm.generate([ChatTurn("user", "again")])
    assert len(completions.calls) == 1
    assert "max_completion_tokens" in completions.calls[0]


@pytest.mark.asyncio
async def test_drops_temperature_and_caches():
    llm, completions = _make_llm(reject={"temperature"})
    out = await llm.generate([ChatTurn("user", "yo")])
    assert out == "hi"
    assert "temperature" in completions.calls[0]
    assert "temperature" not in completions.calls[1]
    assert llm._send_temperature is False

    # Cached — subsequent calls never send temperature again.
    completions.calls.clear()
    await llm.generate([ChatTurn("user", "again")])
    assert len(completions.calls) == 1
    assert "temperature" not in completions.calls[0]


@pytest.mark.asyncio
async def test_recovers_when_both_params_rejected():
    llm, completions = _make_llm(reject={"max_tokens", "temperature"})
    out = await llm.generate([ChatTurn("user", "yo")])
    assert out == "hi"
    # Converges within the bounded retry loop despite two separate 400s.
    assert llm._token_param == "max_completion_tokens"
    assert llm._send_temperature is False
    final = completions.calls[-1]
    assert "max_completion_tokens" in final and "temperature" not in final


@pytest.mark.asyncio
async def test_non_param_400_propagates():
    class _OtherError(Exception):
        status_code = 400
        message = "You exceeded your quota."

    llm, completions = _make_llm(reject=None)

    async def _boom(**kwargs):
        completions.calls.append(kwargs)
        raise _OtherError()

    llm._client.chat.completions.create = _boom
    with pytest.raises(_OtherError):
        await llm.generate([ChatTurn("user", "yo")])
    assert len(completions.calls) == 1  # no pointless retries


@pytest.mark.asyncio
async def test_happy_path_keeps_max_tokens():
    llm, completions = _make_llm(reject=None)
    await llm.generate([ChatTurn("user", "yo")])
    assert len(completions.calls) == 1
    assert "max_tokens" in completions.calls[0]
    assert "temperature" in completions.calls[0]
    assert llm._token_param == "max_tokens"
