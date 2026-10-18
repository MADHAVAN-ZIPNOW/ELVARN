-- Stage 2: hourly e-mail bookkeeping, reminders (JARVIS Call Mode) and a small job log. Text only, tiny rows.
ALTER TABLE memory_settings
    ADD COLUMN last_email_at   timestamptz,
    ADD COLUMN email_sent_day  date,
    ADD COLUMN email_sent_count integer NOT NULL DEFAULT 0;

CREATE TABLE reminders (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id   uuid        NOT NULL,
    text       text        NOT NULL CHECK (length(text) BETWEEN 1 AND 300),
    remind_at  timestamptz NOT NULL,
    status     text        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'done')),
    created_at timestamptz NOT NULL DEFAULT now(),
    sent_at    timestamptz
);
CREATE INDEX idx_rem_due ON reminders (owner_id, remind_at) WHERE status = 'pending';

CREATE TABLE job_log (
    id       bigserial PRIMARY KEY,
    owner_id uuid        NOT NULL,
    job      text        NOT NULL,
    at       timestamptz NOT NULL DEFAULT now(),
    ok       boolean     NOT NULL,
    detail   text        NOT NULL DEFAULT '' CHECK (length(detail) <= 500)
);
