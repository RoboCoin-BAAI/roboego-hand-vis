# Action cleaning

Edit nonverb_rules.json: delete keys are the exact labels to remove, values are explanations. replace maps confirmed synonyms to canonical labels. Replaced lists are deduplicated in original order.
Run after manually confirming the list:

```bash
python -m roboego_hand_vis.annotation_statistics.clean_actions --input-root /share/project/liuyou/humancoin/data/guanglun_depth
```

Directly updates only actions.vocabulary. subtask[].action and all other field values are preserved. Empty lists are allowed. Other field values are preserved. Modified JSON uses two-space indentation. No logs, backups or preview files. Prints counts and errors only.

Current list: continuously, repeatedly, neatly, carefully, finally, completely, fully, thoroughly. All other labels remain unchanged, including upright and hang.

Confirmed synonyms: inspect/verify -> check; grab -> grasp; join -> connect; fasten -> secure. No other synonym groups are applied.

Additional confirmed synonyms: put -> place; shut -> close; tidy -> organize; gather -> collect.

Normalization runs before deletion and replacement: lowercase, then remove surrounding whitespace and Unicode punctuation (including commas and quotes). Internal punctuation is preserved: pick_up and fine-tune remain intact. Empty labels produced by normalization are removed. Symbols such as + are not treated as punctuation. Only actions.vocabulary uses these rules and stable deduplication.

Uses fixed batch_*/<session>/annotation/episode_000000_structured.json discovery. Supports a single batch/session/annotation root as well. --rules accepts another confirmed deletion list.

Confirmed deletion of non-specific labels: continue, repeat, keep, start, complete, finish. The previous finish -> complete mapping is removed.

Additional confirmed deletions: perform, use, work, work_on, manipulate, handle, use_tool, operate_tool, finalize, prepare_place, wait.

Additional confirmed deletions: position_hands, adjust_hands, rest, observe, read, pan_camera, stop_pouring, stop_writing.

Confirmed naming mappings: pick_up -> pick; return_tool -> return; remove_cap -> uncap; close_book -> close; close_lid -> close; open_lid -> open; turn_pages -> turn_page.
