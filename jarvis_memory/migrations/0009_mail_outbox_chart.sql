-- The 3D chart picture that goes with a queued team e-mail (the scheduler attaches it).
ALTER TABLE public.mail_outbox ADD COLUMN chart_png bytea;
