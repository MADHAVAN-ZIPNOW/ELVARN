-- Stage 5: the nightly Daily Review (one short text row per day; ~1 KB a day, about 350 KB a year).
ALTER TABLE memory_settings
    ADD COLUMN review_enabled boolean NOT NULL DEFAULT true,
    ADD COLUMN review_hour    integer NOT NULL DEFAULT 21 CHECK (review_hour BETWEEN 0 AND 23),
    ADD COLUMN review_minute  integer NOT NULL DEFAULT 30 CHECK (review_minute BETWEEN 0 AND 59),
    ADD COLUMN last_review_on date;

CREATE TABLE daily_reviews (
    owner_id   uuid        NOT NULL,
    day        date        NOT NULL,
    summary    text        NOT NULL CHECK (length(summary) <= 4000),
    stats      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    details    jsonb       NOT NULL DEFAULT '{}'::jsonb,     -- done tasks, open tasks, reminders: what the review was built from
    emailed    boolean     NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id, day)
);
