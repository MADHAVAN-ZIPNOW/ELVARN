-- Stage 6: check-in e-mails go out at fixed times of day (default 09:00, 12:00, 17:00) instead of every hour.
ALTER TABLE memory_settings
    ADD COLUMN email_times text NOT NULL DEFAULT '09:00,12:00,17:00' CHECK (length(email_times) <= 80);
