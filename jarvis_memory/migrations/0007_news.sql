-- Stage 7: live AI / quantum / space news notifications (tiny: one row per story already sent, pruned after 14 days).
ALTER TABLE memory_settings
    ADD COLUMN news_enabled          boolean     NOT NULL DEFAULT true,
    ADD COLUMN news_topics           text        NOT NULL DEFAULT 'ai,quantum,space' CHECK (length(news_topics) <= 60),
    ADD COLUMN news_interval_minutes integer     NOT NULL DEFAULT 120 CHECK (news_interval_minutes BETWEEN 15 AND 1440),
    ADD COLUMN max_news_per_day      integer     NOT NULL DEFAULT 6 CHECK (max_news_per_day BETWEEN 0 AND 50),
    ADD COLUMN last_news_at          timestamptz,
    ADD COLUMN news_sent_day         date,
    ADD COLUMN news_sent_count       integer     NOT NULL DEFAULT 0;

CREATE TABLE public.news_seen (
    owner_id  uuid        NOT NULL,
    link_hash text        NOT NULL,
    seen_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id, link_hash)
);
