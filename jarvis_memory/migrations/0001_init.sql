-- Jarvis Memory, migration 0001 (Stage 1).
--
-- Designed to stay small on Neon's free Postgres: ONE row per memory. Tags / projects / people / next steps are
-- Postgres arrays and tasks are a small JSONB array on the same row (no child tables, no per-row UUID fan-out).
-- Text is never stored twice: `content` is always the exact original; `raw_transcript` is only set for voice
-- memories (where `content` is the user-edited version of what speech recognition produced).
-- Estimated cost: ~2-4 KB per memory including indexes -> hundreds of thousands of memories per GB.

-- ---------------------------------------------------------------------------------------- search helpers
-- array_to_string() is only STABLE, which blocks it from index expressions; these wrappers are safe to mark
-- IMMUTABLE because they only ever join plain text.
CREATE OR REPLACE FUNCTION jm_join(a text[]) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT coalesce(array_to_string(a, ' '), '') $$;

CREATE OR REPLACE FUNCTION jm_tasks_text(t jsonb) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT coalesce(string_agg(e ->> 'task', ' '), '') FROM jsonb_array_elements(coalesce(t, '[]'::jsonb)) e $$;

-- The text that full-text search looks at (the same expression is used by the index and by every query).
CREATE OR REPLACE FUNCTION jm_search_text(
    title text, content text, raw_transcript text, summary text,
    tags text[], projects text[], people text[], next_steps text[], tasks jsonb
) RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT coalesce(title, '') || ' ' || coalesce(content, '') || ' ' || coalesce(raw_transcript, '') || ' ' ||
          coalesce(summary, '') || ' ' || public.jm_join(tags) || ' ' || public.jm_join(projects) || ' ' || public.jm_join(people) || ' ' ||
          public.jm_join(next_steps) || ' ' || public.jm_tasks_text(tasks) $$;

CREATE OR REPLACE FUNCTION jm_tsv(
    title text, content text, raw_transcript text, summary text,
    tags text[], projects text[], people text[], next_steps text[], tasks jsonb
) RETURNS tsvector LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT to_tsvector('english'::regconfig,
                      public.jm_search_text(title, content, raw_transcript, summary, tags, projects, people, next_steps, tasks)) $$;

-- ---------------------------------------------------------------------------------------- memories
CREATE TABLE memory_entries (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id              uuid        NOT NULL,
    source                text        NOT NULL CHECK (source IN ('manual', 'voice', 'email', 'daily_review')),
    title                 text        NOT NULL DEFAULT '',
    content               text        NOT NULL,          -- exact original text (voice: the final, user-edited transcript)
    raw_transcript        text,                           -- voice only: exactly what speech recognition produced
    summary               text        NOT NULL DEFAULT '', -- AI generated, editable, kept apart from the original
    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now(),
    start_time            timestamptz,
    end_time              timestamptz,
    timezone              text        NOT NULL DEFAULT 'Asia/Kolkata',
    thread_id             text,                           -- links a reply to the hourly request it answers (Stage 2)
    external_reference_id text,                           -- e.g. an email Message-ID: makes inbound handling idempotent
    sentiment             text CHECK (sentiment IN ('positive', 'neutral', 'negative', 'mixed')),
    tags                  text[]      NOT NULL DEFAULT '{}',
    projects              text[]      NOT NULL DEFAULT '{}',
    people                text[]      NOT NULL DEFAULT '{}',
    next_steps            text[]      NOT NULL DEFAULT '{}',
    tasks                 jsonb       NOT NULL DEFAULT '[]'::jsonb,   -- [{"task": "...", "status": "pending|done|blocked", "due_at": null}]
    metadata_json         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    deleted_at            timestamptz,
    CONSTRAINT jm_size_guard CHECK (length(content) <= 20000 AND length(coalesce(raw_transcript, '')) <= 20000)
);

-- Lists and the default "newest first" view.
CREATE INDEX idx_me_owner_created ON memory_entries (owner_id, created_at DESC) WHERE deleted_at IS NULL;
-- Source filter + timeline.
CREATE INDEX idx_me_owner_source ON memory_entries (owner_id, source, created_at DESC) WHERE deleted_at IS NULL;
-- Full-text search: one GIN expression index, no stored tsvector column (saves a copy of every row's text).
CREATE INDEX idx_me_fts ON memory_entries USING gin (
    jm_tsv(title, content, raw_transcript, summary, tags, projects, people, next_steps, tasks)
) WHERE deleted_at IS NULL;
-- Thread links and idempotency keys are sparse, so these are partial (almost no space).
CREATE INDEX idx_me_thread ON memory_entries (thread_id) WHERE thread_id IS NOT NULL;
CREATE UNIQUE INDEX uq_me_external_ref ON memory_entries (owner_id, external_reference_id) WHERE external_reference_id IS NOT NULL;
-- Task-status filter ("pending", "blocked"): only rows that have tasks are indexed.
CREATE INDEX idx_me_tasks ON memory_entries USING gin (tasks jsonb_path_ops) WHERE tasks <> '[]'::jsonb;

-- ---------------------------------------------------------------------------------------- settings (one row per owner)
CREATE TABLE memory_settings (
    owner_id                  uuid PRIMARY KEY,
    owner_name                text    NOT NULL DEFAULT '',
    owner_email               text    NOT NULL DEFAULT '',
    timezone                  text    NOT NULL DEFAULT 'Asia/Kolkata',
    -- used by later stages; stored now so the schema never needs a rewrite
    email_enabled             boolean NOT NULL DEFAULT false,
    email_interval_minutes    integer NOT NULL DEFAULT 60 CHECK (email_interval_minutes BETWEEN 15 AND 1440),
    push_enabled              boolean NOT NULL DEFAULT false,
    push_interval_minutes     integer NOT NULL DEFAULT 60 CHECK (push_interval_minutes BETWEEN 15 AND 1440),
    quiet_start_hour          integer NOT NULL DEFAULT 22 CHECK (quiet_start_hour BETWEEN 0 AND 23),
    quiet_end_hour            integer NOT NULL DEFAULT 9  CHECK (quiet_end_hour BETWEEN 0 AND 23),
    max_emails_per_day        integer NOT NULL DEFAULT 20 CHECK (max_emails_per_day BETWEEN 0 AND 100),
    max_notifications_per_day integer NOT NULL DEFAULT 20 CHECK (max_notifications_per_day BETWEEN 0 AND 100),
    -- Stage 1: protects the free database. Writes are refused above this size (reads and export keep working).
    db_soft_limit_mb          integer NOT NULL DEFAULT 400 CHECK (db_soft_limit_mb BETWEEN 50 AND 5000),
    updated_at                timestamptz NOT NULL DEFAULT now()
);
