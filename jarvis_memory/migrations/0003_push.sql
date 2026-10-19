-- Stage 3: phone push (Firebase). One small row per phone, plus counters on the settings row.
CREATE TABLE devices (
    token      text PRIMARY KEY CHECK (length(token) BETWEEN 20 AND 400),
    owner_id   uuid        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    seen_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_devices_owner ON devices (owner_id);

ALTER TABLE memory_settings
    ADD COLUMN last_call_at          timestamptz,      -- hourly "what did you do?" call
    ADD COLUMN last_task_push_at     timestamptz,      -- 30-minute open-task nudge
    ADD COLUMN push_sent_day         date,
    ADD COLUMN push_sent_count       integer NOT NULL DEFAULT 0;
