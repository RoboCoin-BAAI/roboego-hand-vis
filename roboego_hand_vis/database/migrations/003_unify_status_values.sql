-- Unify every download/stage status constraint to five values.
-- Execute in one transaction; safe to repeat, preserving existing records.

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_status_valid,
    ADD CONSTRAINT l1data_status_valid
        CHECK (download_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l2data
    DROP CONSTRAINT IF EXISTS l2data_status_valid,
    ADD CONSTRAINT l2data_status_valid
        CHECK (download_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l3data
    DROP CONSTRAINT IF EXISTS l3data_status_valid,
    ADD CONSTRAINT l3data_status_valid
        CHECK (download_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_raw_headband_integrity_status_valid,
    ADD CONSTRAINT l1data_raw_headband_integrity_status_valid
        CHECK (raw_headband_integrity_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_video_simple_quality_status_valid,
    ADD CONSTRAINT l1data_video_simple_quality_status_valid
        CHECK (video_simple_quality_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_detect_hand_status_valid,
    ADD CONSTRAINT l1data_detect_hand_status_valid
        CHECK (detect_hand_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_stereo_rectification_status_valid,
    ADD CONSTRAINT l1data_stereo_rectification_status_valid
        CHECK (stereo_rectification_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_qwen_annotation_status_valid,
    ADD CONSTRAINT l1data_qwen_annotation_status_valid
        CHECK (qwen_annotation_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_post_annotation_actions_status_valid,
    ADD CONSTRAINT l1data_post_annotation_actions_status_valid
        CHECK (post_annotation_actions_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_stereo_depth_gate_status_valid,
    ADD CONSTRAINT l1data_stereo_depth_gate_status_valid
        CHECK (stereo_depth_gate_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_fast_foundation_depth_status_valid,
    ADD CONSTRAINT l1data_fast_foundation_depth_status_valid
        CHECK (fast_foundation_depth_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_hand_status_valid,
    ADD CONSTRAINT l1data_hand_status_valid
        CHECK (hand_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_archive_depth_cache_status_valid,
    ADD CONSTRAINT l1data_archive_depth_cache_status_valid
        CHECK (archive_depth_cache_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_vio_status_valid,
    ADD CONSTRAINT l1data_vio_status_valid
        CHECK (vio_status IN ('pending', 'processing', 'completed', 'failed', 'block'));

ALTER TABLE l1data
    DROP CONSTRAINT IF EXISTS l1data_video_privacy_blur_status_valid,
    ADD CONSTRAINT l1data_video_privacy_blur_status_valid
        CHECK (video_privacy_blur_status IN ('pending', 'processing', 'completed', 'failed', 'block'));
