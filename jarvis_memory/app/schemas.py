"""Request/response models. Lists are cleaned and capped so one memory can never grow into a large row."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Source = Literal["manual", "voice", "email", "daily_review"]
TaskStatus = Literal["pending", "done", "blocked"]
Sentiment = Literal["positive", "neutral", "negative", "mixed"]


def clean_list(items: Optional[list[str]], max_items: int, max_len: int, lower: bool = False) -> list[str]:
    """Trim, drop empties, de-duplicate (case-insensitively, keeping the first spelling) and cap."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in items or []:
        item = " ".join(str(raw).split())[:max_len].strip()
        if lower:
            item = item.lower()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= max_items:
            break
    return out


class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str = Field(min_length=1, max_length=300)
    status: TaskStatus = "pending"
    due_at: Optional[datetime] = None

    @field_validator("task")
    @classmethod
    def _squash(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("task must not be empty")
        return v


class _Lists(BaseModel):
    tags: Optional[list[str]] = None
    projects: Optional[list[str]] = None
    people: Optional[list[str]] = None
    next_steps: Optional[list[str]] = None
    tasks: Optional[list[Task]] = Field(None, max_length=30)

    @field_validator("tags")
    @classmethod
    def _tags(cls, v):
        return None if v is None else clean_list(v, 20, 40, lower=True)

    @field_validator("projects")
    @classmethod
    def _projects(cls, v):
        return None if v is None else clean_list(v, 10, 80)

    @field_validator("people")
    @classmethod
    def _people(cls, v):
        return None if v is None else clean_list(v, 20, 80)

    @field_validator("next_steps")
    @classmethod
    def _steps(cls, v):
        return None if v is None else clean_list(v, 20, 300)


class MemoryCreate(_Lists):
    """A new memory. ``content`` is stored exactly as sent (it is the original and is never rewritten)."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=20000)
    source: Literal["manual", "voice"] = "manual"
    #: Voice only: exactly what speech recognition returned, before the user edited it into ``content``.
    raw_transcript: Optional[str] = Field(None, max_length=20000)
    title: Optional[str] = Field(None, max_length=200)
    summary: Optional[str] = Field(None, max_length=2000)
    sentiment: Optional[Sentiment] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    thread_id: Optional[str] = Field(None, max_length=120)
    extract: bool = True

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class MemoryUpdate(_Lists):
    """Editable fields. The original ``content`` / ``raw_transcript`` / ``source`` can never be changed."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(None, max_length=200)
    summary: Optional[str] = Field(None, max_length=2000)
    sentiment: Optional[Sentiment] = None
    clear_sentiment: bool = False


class MemoryOut(BaseModel):
    id: str
    source: Source
    title: str
    content: str
    raw_transcript: Optional[str]
    summary: str
    created_at: datetime
    updated_at: datetime
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    timezone: str
    thread_id: Optional[str]
    external_reference_id: Optional[str]
    sentiment: Optional[Sentiment]
    tags: list[str]
    projects: list[str]
    people: list[str]
    next_steps: list[str]
    tasks: list[dict[str, Any]]
    metadata_json: dict[str, Any]
    attachments: list[dict[str, Any]] = Field(default_factory=list)  # filled in by Stage 4 (media)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field("", max_length=300)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sources: Optional[list[Source]] = Field(None, max_length=4)
    project: Optional[str] = Field(None, max_length=80)
    person: Optional[str] = Field(None, max_length=80)
    tag: Optional[str] = Field(None, max_length=40)
    task_status: Optional[TaskStatus] = None
    sentiment: Optional[Sentiment] = None
    has_attachment: Optional[bool] = None
    has_transcript: Optional[bool] = None
    sort: Literal["newest", "oldest", "relevance"] = "newest"
    page: int = Field(1, ge=1, le=10000)
    page_size: int = Field(20, ge=1, le=50)


class SearchHit(BaseModel):
    memory: MemoryOut
    #: Plain text where matches are wrapped in \u0002 ... \u0003 (render them as highlights; never as HTML).
    snippet: str
    rank: float = 0.0


class SearchResponse(BaseModel):
    items: list[SearchHit]
    total: int
    page: int
    page_size: int
    has_more: bool


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_name: Optional[str] = Field(None, max_length=80)
    owner_email: Optional[str] = Field(None, max_length=200)
    timezone: Optional[str] = Field(None, max_length=60)
    email_enabled: Optional[bool] = None
    email_interval_minutes: Optional[int] = Field(None, ge=15, le=1440)
    push_enabled: Optional[bool] = None
    push_interval_minutes: Optional[int] = Field(None, ge=15, le=1440)
    quiet_start_hour: Optional[int] = Field(None, ge=0, le=23)
    quiet_end_hour: Optional[int] = Field(None, ge=0, le=23)
    max_emails_per_day: Optional[int] = Field(None, ge=0, le=100)
    max_notifications_per_day: Optional[int] = Field(None, ge=0, le=100)
    db_soft_limit_mb: Optional[int] = Field(None, ge=50, le=5000)

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v):
        if v is None:
            return v
        from zoneinfo import ZoneInfo

        try:
            ZoneInfo(v)
        except Exception as exc:
            raise ValueError("unknown timezone") from exc
        return v


class ReminderCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=300)
    remind_at: datetime

    @field_validator("text")
    @classmethod
    def _squash(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("text must not be empty")
        return v


class SnoozeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minutes: int = Field(10, ge=1, le=1440)
