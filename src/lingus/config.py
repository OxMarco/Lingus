"""Typed configuration: merges config.yaml (structure) with .env (secrets).

`Settings.load()` is the single entrypoint. Everything else reads the returned
`Settings` object; nothing reads os.environ or the YAML directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- Secrets (from environment / .env) ---------------------------------------
class Secrets(BaseSettings):
    """Secrets pulled from the environment. Never serialized back to disk."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    youtube_client_secrets: str = Field(default="", alias="YOUTUBE_CLIENT_SECRETS")
    youtube_token_path: str = Field(default=".youtube_token.json", alias="YOUTUBE_TOKEN_PATH")
    config_path: str = Field(default="config.yaml", alias="LINGUS_CONFIG")


# --- Structured config (from config.yaml) ------------------------------------
class FileReplayConfig(BaseModel):
    segment_path: str = ""


class YouTubeConfig(BaseModel):
    video_id: str = ""
    # Read the live chat (keyless InnerTube reader). Off = speech-only observe.
    chat_enabled: bool = True


class ASRConfig(BaseModel):
    backend: str = "faster_whisper"
    model_size: str = "turbo"
    device: str = "auto"
    # Most streams are single-language; pinning skips Whisper's per-window
    # detection (no flapping to the wrong language) and trims latency. Default to
    # English; override per-deployment in config.yaml or per-run with --language
    # (use --language auto to restore per-window autodetection).
    language: str | None = "en"
    # Whisper pads every chunk to 30s internally, so a 5s window pays the full
    # encoder cost for a sliver of audio (~6x waste). 10s windows cut real-time
    # factor roughly in half — enough headroom to run turbo on a strong local
    # CPU — and give the model more context, at the cost of +5s latency. Drop to
    # 5s only if you need minimum latency and run a small model.
    window_seconds: float = 10.0


class AudioGateConfig(BaseModel):
    # "spectral" = lightweight local speech/music gate; "hf_ast" = Hugging Face
    # AudioSet classifier; "none" disables pre-ASR gating.
    backend: str = "spectral"
    window_seconds: float = Field(default=2.0, gt=0.0)
    speech_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    music_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    silence_rms: float = Field(default=0.003, ge=0.0)
    replacement_silence_seconds: float = Field(default=0.5, ge=0.0)
    hf_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    hf_top_k: int = Field(default=12, ge=1)
    hf_cache_dir: str = ".lingus/hf/hub"
    hf_local_files_only: bool = True


class LLMConfig(BaseModel):
    backend: str = "openai_compat"
    model: str = "gpt-5.5"
    temperature: float = 0.9
    max_tokens: int = 120


class VLMConfig(BaseModel):
    # "mlx_vlm" = local Apple Silicon VLM via mlx-vlm; "local_cv" = cheap local
    # frame analysis fallback; "none" disables live video.
    backend: str = "mlx_vlm"
    model: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    max_tokens: int = Field(default=180, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    fallback_to_local_cv: bool = True
    # Phase 4 frame gate: how different a sampled RGB frame must be from the
    # last accepted frame before local analysis runs again.
    frame_diff_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    frame_min_interval_seconds: float = Field(default=3.0, ge=0.0)
    max_sample_pixels: int = Field(default=4096, ge=1)
    brightness_change_threshold: float = Field(default=0.16, ge=0.0, le=1.0)
    contrast_change_threshold: float = Field(default=0.10, ge=0.0, le=1.0)


class ModerationConfig(BaseModel):
    # "regex" = deterministic filter (see lingus/safety.py); "none" disables the
    # gate (offline replay tuning only — real posting must keep it on).
    backend: str = "regex"
    # Per-deployment additions to the built-in denylist (raw regexes).
    extra_patterns: list[str] = Field(default_factory=list)
    # Spam-shape detection (link floods, scam CTAs). Independent of the slur
    # denylist so a link-friendly persona isn't muzzled.
    check_spam: bool = True


class ModelsConfig(BaseModel):
    asr: ASRConfig = Field(default_factory=ASRConfig)
    audio_gate: AudioGateConfig = Field(default_factory=AudioGateConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    moderation: ModerationConfig = Field(default_factory=ModerationConfig)


class PersonaConfig(BaseModel):
    path: str = "src/lingus/persona/personas/default.yaml"


class ArbiterConfig(BaseModel):
    fire_threshold: float = Field(default=1.0, ge=0.0)
    cooldown_seconds: float = Field(default=20.0, gt=0.0)  # post-speak bar decay
    cooldown_bump: float = Field(default=1.0, ge=0.0)  # bar jump right after speaking
    lull_after_seconds: float = Field(default=25.0, ge=0.0)  # silence before lull builds
    mood_threshold_gain: float = Field(default=0.3, ge=0.0)  # mood strength
    weights: dict[str, float] = Field(default_factory=dict)


class ChatTrendsConfig(BaseModel):
    """Pile-on / "follow the trend" detector (see lingus/chat_trends.py)."""

    enabled: bool = True
    window_seconds: float = Field(default=12.0, gt=0.0)  # how far back a wave counts
    min_senders: int = Field(default=4, ge=1)  # distinct authors before it's a trend
    min_fraction: float = Field(default=0.35, gt=0.0, le=1.0)  # share of windowed chat
    max_token_chars: int = Field(default=24, gt=0)  # longer lines aren't pile-ons
    follow_probability: float = Field(default=0.6, ge=0.0, le=1.0)  # join rate = temperament
    fatigue_seconds: float = Field(default=90.0, ge=0.0)  # don't echo the same bit again
    cooldown_seconds: float = Field(default=20.0, ge=0.0)  # min gap between any two follows


class MemoryConfig(BaseModel):
    """Self-memory dedup + bit-fatigue (see lingus/memory/repetition.py)."""

    # Token-Jaccard overlap with a recent bot message that counts as a repeat.
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # How long a catchphrase stays "spent" after the bot leans on it.
    fatigue_seconds: float = Field(default=180.0, ge=0.0)
    # Episodic memory: fold evicted transcript lines into a "stream so far" digest.
    episodic_enabled: bool = True
    episodic_batch_lines: int = Field(default=8, ge=1)  # summarize once this many pile up
    episodic_max_chars: int = Field(default=800, gt=0)  # cap on the running narrative
    episodic_path: str = ".lingus/episodes.json"  # per-stream summaries across runs
    episodic_max_entries: int = Field(default=20, ge=1)
    episodic_top_k: int = Field(default=3, ge=0)  # prior summaries surfaced into context
    # Semantic memory: durable facts persisted across streams.
    semantic_enabled: bool = True
    semantic_path: str = ".lingus/semantic.json"  # where facts persist between runs
    semantic_max_facts: int = Field(default=50, ge=1)
    semantic_top_k: int = Field(default=5, ge=1)  # facts surfaced into context


class WebSearchConfig(BaseModel):
    """Live web search used by the cold-start channel researcher."""

    # "duckduckgo" = keyless DDG scrape (needs the [research] extra); "none"
    # disables web search (researcher then falls back to yt-dlp metadata only).
    backend: str = "duckduckgo"
    max_results: int = Field(default=6, ge=1)  # snippets kept per query
    max_queries: int = Field(default=4, ge=1)  # distinct searches per channel


class ResearchConfig(BaseModel):
    """Cold-start channel research: profile the streamer BEFORE the loop starts
    and seed the durable (semantic) memory, so the bot walks in already knowing
    the channel instead of learning it from scratch each stream."""

    enabled: bool = True
    # For non-YouTube platforms (twitch/file_replay) there's no yt-dlp identity to
    # resolve — set the channel name here to research it anyway. Ignored on
    # YouTube, where identity comes from the video's channel.
    channel: str = ""
    # A known channel is researched once; re-research only after this many days
    # (channels evolve, but not every stream). 0 = always re-research.
    refresh_days: float = Field(default=14.0, ge=0.0)
    max_facts: int = Field(default=12, ge=1)  # cap on facts seeded into memory
    cache_dir: str = ".lingus/research"  # per-channel profile cache
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class OutputConfig(BaseModel):
    max_chars: int = Field(default=200, gt=0)  # hard length cap
    min_seconds_between_posts: float = Field(default=8.0, ge=0.0)  # hard floor between posts
    # Token bucket: sustained rate + how many posts may bunch up as burst.
    posts_per_minute: float = Field(default=6.0, gt=0.0)
    burst: int = Field(default=2, ge=1)
    # On an over-length reply, try one tighter regeneration before truncating.
    regenerate_on_overflow: bool = True
    # Optional flavor delay. Keep off by default so live replies stay responsive.
    typing_enabled: bool = False
    typing_cps: float = Field(default=15.0, gt=0.0)  # characters "typed" per second
    typing_base_seconds: float = Field(default=0.4, ge=0.0)  # reaction beat before typing
    typing_min_seconds: float = Field(default=0.8, ge=0.0)
    typing_max_seconds: float = Field(default=7.0, gt=0.0)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    # `json` in config.yaml maps here; aliased to avoid shadowing BaseModel.json.
    as_json: bool = Field(default=False, alias="json")


class Settings(BaseModel):
    platform: Literal["youtube", "twitch", "file_replay"] = "file_replay"
    file_replay: FileReplayConfig = Field(default_factory=FileReplayConfig)
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    arbiter: ArbiterConfig = Field(default_factory=ArbiterConfig)
    chat_trends: ChatTrendsConfig = Field(default_factory=ChatTrendsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Populated from Secrets at load time, not from YAML.
    secrets: Secrets = Field(default_factory=Secrets)

    @classmethod
    def load(cls, config_path: str | os.PathLike[str] | None = None) -> Settings:
        secrets = Secrets()
        path = Path(config_path or secrets.config_path)
        data: dict[str, Any] = {}
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
        elif config_path is not None or path != Path("config.yaml"):
            raise FileNotFoundError(f"config file not found: {path}")
        settings = cls.model_validate(data)
        settings.secrets = secrets
        return settings
