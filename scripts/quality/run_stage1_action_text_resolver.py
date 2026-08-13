#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.stage1_action_text_resolver import (
    call_qwen_text_resolver,
    merge_text_resolver_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-only Qwen fallback for uncertain Stage 1 action bindings.")
    parser.add_argument("--review_json", required=True)
    parser.add_argument("--input_actions_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resolved_json", default="")
    parser.add_argument("--raw_output_txt", default="")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--timeout_s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_payload = json.loads(Path(args.review_json).read_text(encoding="utf-8"))
    actions_payload = json.loads(Path(args.input_actions_json).read_text(encoding="utf-8"))

    parsed = call_qwen_text_resolver(
        review_payload=review_payload,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_s=args.timeout_s,
    )
    raw_output_txt = Path(args.raw_output_txt) if args.raw_output_txt else Path(args.output_json).with_suffix(".raw.txt")
    raw_output_txt.parent.mkdir(parents=True, exist_ok=True)
    raw_output_txt.write_text(str(parsed.get("raw_response", "")), encoding="utf-8")

    resolved_json = Path(args.resolved_json) if args.resolved_json else Path(args.output_json).with_suffix(".resolved.json")
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    if parsed.get("status") != "ok":
        raise RuntimeError(f"Qwen text resolver failed: {parsed.get('status')}")

    merged = merge_text_resolver_results(actions_payload, parsed)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "resolved_json": str(resolved_json),
                "raw_output_txt": str(raw_output_txt),
                "text_resolver_frame_count": merged.get("text_resolver_frame_count", 0),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
