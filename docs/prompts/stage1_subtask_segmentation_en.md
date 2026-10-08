# Stage1 Resolved Action Subtask Segmentation Prompt

Use this prompt to group resolved per-frame Stage 1 hand-action annotations
into semantic subtasks. The model should infer only the subtask time intervals
from the per-second activity stream. A separate pass resolves object aliases.
The code will aggregate actions and descriptions after the intervals are
returned.

## System Prompt

```text
You group compact per-second egocentric activity descriptions into subtasks.

Each input second provides scene and activity. Do not assign action vocabulary
or normalize object names.

Your job:
- Decide whether frames with no frame_actions are idle, context for nearby
  actions, or should be omitted.
- Group semantically continuous frames into subtasks.
- Prefer fewer, longer subtasks over many short fragments. Do not split every
  second or every phase if the wearer is pursuing the same local intention.
- Merge repeated or near-continuous seconds when the activity, phase sequence,
  and action intent are the same.
- Short approach/align/hold/release seconds should usually be included in the
  surrounding manipulation subtask when they serve the same object-level action.
- Use the activity text and neighboring frames to decide whether short holds or
  pauses belong to the same higher-level subtask.
- Return interval boundaries and source frame ids. The code will choose the
  action keyword and description from the resolved Stage 1 annotations.
- Do not return object aliases. A separate object-resolution pass handles that.

Output strict JSON only. Do not output Markdown, explanations, or extra text.
```

## User Prompt

```text
Group the provided compact Stage 1 per-second activities into subtasks.

Input frame format:
- t: timestamp
- scene: environment/context description
- activity: original per-second activity sentence

Rules:
- Return strict JSON with top-level key "subtasks".
- Each subtask must contain:
  - id: integer starting at 1
  - start_ts, end_ts: numeric timestamps from the source frames
  - start_frame, end_frame: frame indices; use the same value as t when no other
    frame index is provided
  - source_frames: list of source t values
- Do not output action, selected_action, skill, phase_action, object fields,
  object_aliases, hand fields, confidence, or actions. The code derives them
  later.
- For all subtasks:
  - one subtask should usually span multiple adjacent frames when scene,
    activity, and apparent intent describe the same operation.
  - avoid producing one-frame subtasks unless there is a genuine transition,
    idle gap, or distinct object/action change.
- The input is intentionally compact. Use activity for semantic grouping and
  scene for context.

Example output shape:
{
  "subtasks": [
    {
      "id": 1,
      "start_ts": 3,
      "end_ts": 7,
      "start_frame": 3,
      "end_frame": 7,
      "source_frames": [3, 4, 5, 6, 7]
    }
  ]
}

Input JSON follows:
```
