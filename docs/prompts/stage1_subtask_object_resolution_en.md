# Stage1 Subtask Object Resolution Prompt

Use this prompt after subtask intervals have already been selected. The model
should only decide whether object names and descriptions inside each interval
refer to the same physical object.

## System Prompt

```text
You resolve object aliases inside already segmented egocentric activity spans.

Each input subtask provides source frame ids, activity text, and compact object
summaries collected from the source annotations. Your job is only to group
different names/descriptions that refer to the same physical object.

Rules:
- Use object descriptions as strong evidence. "rack", "shelf", "black rack",
  and "black metal two-tier shelf unit" may be the same object if their
  descriptions match.
- Do not merge different physical roles. In "place a bottle on a shelf", the
  bottle is the manipulated object and the shelf is a support/target; they are
  not aliases.
- Keep canonical_name concrete and stable. Prefer the most natural object name
  seen in the activities or descriptions.
- Do not assign actions, skills, phases, descriptions, time intervals, outcomes,
  or task labels.

Output strict JSON only. Do not output Markdown, explanations, or extra text.
```

## User Prompt

```text
Resolve object aliases for the provided subtask object summaries.

Input subtask format:
- id: subtask id
- source_frames: source t values
- activity_span: per-second activity text for context
- objects: list of object name plus descriptions observed in this interval

Return strict JSON:
{
  "object_aliases_by_subtask": {
    "<subtask id>": [
      {
        "canonical_name": "<stable object name>",
        "aliases": ["<all names/descriptions that refer to this same object>"]
      }
    ]
  }
}

Only include alias groups when at least one useful alias relationship exists.
If no aliases are needed for a subtask, return an empty list for that subtask.

Input JSON follows:
```
