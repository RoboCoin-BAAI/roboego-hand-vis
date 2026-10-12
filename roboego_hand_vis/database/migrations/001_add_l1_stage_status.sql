-- Add the 12 L1 stage-status columns to an existing four-column l1data table.
-- Existing data is preserved; new status values default to pending.
-- Re-running this migration skips columns that already exist.
ALTER TABLE l1data
    ADD COLUMN IF NOT EXISTS raw_headband_integrity_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_raw_headband_integrity_status_valid
        CHECK (raw_headband_integrity_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS video_simple_quality_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_video_simple_quality_status_valid
        CHECK (video_simple_quality_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS detect_hand_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_detect_hand_status_valid
        CHECK (detect_hand_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS stereo_rectification_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_stereo_rectification_status_valid
        CHECK (stereo_rectification_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS qwen_annotation_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_qwen_annotation_status_valid
        CHECK (qwen_annotation_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS post_annotation_actions_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_post_annotation_actions_status_valid
        CHECK (post_annotation_actions_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS stereo_depth_gate_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_stereo_depth_gate_status_valid
        CHECK (stereo_depth_gate_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS fast_foundation_depth_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_fast_foundation_depth_status_valid
        CHECK (fast_foundation_depth_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS hand_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_hand_status_valid
        CHECK (hand_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS archive_depth_cache_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_archive_depth_cache_status_valid
        CHECK (archive_depth_cache_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS vio_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_vio_status_valid
        CHECK (vio_status IN ('pending', 'processing', 'completed', 'failed')),
    ADD COLUMN IF NOT EXISTS video_privacy_blur_status TEXT NOT NULL DEFAULT 'pending'
        CONSTRAINT l1data_video_privacy_blur_status_valid
        CHECK (video_privacy_blur_status IN ('pending', 'processing', 'completed', 'failed'));
