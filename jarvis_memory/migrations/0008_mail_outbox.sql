-- E-mails the server cannot send itself (Render's free plan blocks SMTP): the scheduler on GitHub Actions sends them and confirms.
CREATE TABLE public.mail_outbox (
    id         bigserial   PRIMARY KEY,
    owner_id   uuid        NOT NULL,
    to_addr    text        NOT NULL,
    subject    text        NOT NULL,
    body_text  text        NOT NULL,
    body_html  text,
    created_at timestamptz NOT NULL DEFAULT now(),
    sent_at    timestamptz
);
CREATE INDEX mail_outbox_pending ON public.mail_outbox (owner_id) WHERE sent_at IS NULL;
