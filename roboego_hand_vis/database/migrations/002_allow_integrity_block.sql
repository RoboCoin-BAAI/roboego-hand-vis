-- Extend only the first quality gate's constraint; preserve all records/statuses.
-- Run in one transaction. Safe to repeat after block records already exist.
ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_raw_headband_integrity_status_valid,
    ADD CONSTRAINT l1data_raw_headband_integrity_status_valid
        CHECK (raw_headband_integrity_status IN
            ('pending', 'processing', 'completed', 'failed', 'block'));
