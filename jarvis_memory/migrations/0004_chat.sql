-- Stage 4: one shared chat for the phone and the desktop (text only, capped sizes).
CREATE TABLE chat_sessions (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id   uuid        NOT NULL,
    title      text        NOT NULL DEFAULT 'New chat' CHECK (length(title) <= 120),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_chat_sessions_owner ON chat_sessions (owner_id, updated_at DESC);

CREATE TABLE chat_messages (
    id         bigserial PRIMARY KEY,
    session_id uuid        NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
    role       text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content    text        NOT NULL CHECK (length(content) <= 12000),
    actions    jsonb       NOT NULL DEFAULT '[]'::jsonb,   -- what JARVIS did for this message: searched memory, added a task, read a page...
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_chat_messages_session ON chat_messages (session_id, id);
