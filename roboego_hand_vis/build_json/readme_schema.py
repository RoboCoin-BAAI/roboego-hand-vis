"""Shared modality layout for compact session readmes."""

MODALITY_GROUPS = (
    'hand_estimation', 'head_trajectory',
    'camera_trajectory', 'eye_gaze', 'imu',
    'hand_trajectory', 'emg', 'eeg', 'glove',
)


def normalize_modalities(modalities):
    result = {group: modalities.get(group, {'streams': []}) for group in MODALITY_GROUPS}
    result.update({group: streams for group, streams in modalities.items() if group not in result})
    return result


def lowercase_values(value):
    """Normalize every string value, preserving keys and non-string types."""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return {key: lowercase_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [lowercase_values(item) for item in value]
    return value
