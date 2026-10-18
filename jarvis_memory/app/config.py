"""Configuration from environment variables. Nothing secret is ever hard-coded or written to a file in the repo."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from functools import lru_cache

# A fixed namespace so the same owner e-mail always maps to the same owner_id (the schema is multi-owner ready).
_OWNER_NAMESPACE = uuid.UUID("6f1d2a0e-3b7c-4c55-9a52-6a1e0c7d4b11")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = ""
    owner_key: str = ""                 # shared secret the dashboard / app send as X-Owner-Key (16+ chars)
    owner_name: str = "Madhavan"
    owner_email: str = ""
    owner_phone: str = ""               # stored for later stages; never logged
    timezone: str = "Asia/Kolkata"
    groq_api_key: str = ""
    llm_model: str = "qwen/qwen3.8-27b"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_timeout_s: float = 20.0
    allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    rate_limit_per_minute: int = 60
    db_soft_limit_mb: int = 400
    ram_limit_mb: int = 400            # Render free = 512 MB hard limit; warn well before it

    @property
    def owner_id(self) -> uuid.UUID:
        return uuid.uuid5(_OWNER_NAMESPACE, (self.owner_email or self.owner_name or "owner").lower())

    @property
    def extraction_enabled(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    origins = tuple(o.strip() for o in _env("JM_ALLOWED_ORIGINS").split(",") if o.strip())
    return Settings(
        database_url=_env("DATABASE_URL"),
        owner_key=_env("JM_OWNER_KEY"),
        owner_name=_env("JARVIS_OWNER_NAME", "Madhavan"),
        owner_email=_env("JARVIS_OWNER_EMAIL"),
        owner_phone=_env("JARVIS_OWNER_PHONE"),
        timezone=_env("JARVIS_TIMEZONE", "Asia/Kolkata"),
        groq_api_key=_env("GROQ_API_KEY"),
        llm_model=_env("JM_LLM_MODEL", "qwen/qwen3.8-27b"),
        llm_base_url=_env("JM_LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        allowed_origins=origins,
        rate_limit_per_minute=_int("JM_RATE_LIMIT_PER_MINUTE", 60),
        db_soft_limit_mb=_int("JM_DB_SOFT_LIMIT_MB", 400),
        ram_limit_mb=_int("JM_RAM_LIMIT_MB", 400),
    )
