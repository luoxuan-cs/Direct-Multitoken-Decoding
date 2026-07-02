#!/usr/bin/env python3
"""Export AM-Thinking raw JSONL to swift-compatible messages-only JSONL.

Strips per-turn metadata (info, more, etc.) so HuggingFace datasets can load
the folder without Arrow schema cast errors.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

_SCRIPT_DIR = Path(__file__).resolve().parent

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
}


def normalize_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    if "conversations" in example and isinstance(example["conversations"], list):
        raw_messages = example["conversations"]
    elif "messages" in example and isinstance(example["messages"], list):
        raw_messages = example["messages"]
    else:
        raise KeyError("Expected 'conversations' or 'messages'.")

    converted = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        if "role" in msg and "content" in msg:
            raw_role, content = msg["role"], msg["content"]
        elif "from" in msg and "value" in msg:
            raw_role, content = msg["from"], msg["value"]
        else:
            continue
        if content is None:
            continue
        raw_role = str(raw_role)
        if raw_role not in ROLE_MAP:
            continue
        converted.append({"role": ROLE_MAP[raw_role], "content": str(content)})
    if not converted:
        raise ValueError("No valid chat turns.")
    return converted


def is_supported(path: str) -> bool:
    base = os.path.basename(path)
    if base in {"dataset_infos.json"}:
        return False
    lower = path.lower()
    return lower.endswith((".jsonl", ".jsonl.gz", ".json", ".json.gz"))


def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as handle:
        if path.lower().endswith((".jsonl", ".jsonl.gz")):
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            data = json.load(handle)
            if isinstance(data, list):
                yield from data
            else:
                yield data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(_SCRIPT_DIR / "raw-am-distilled"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(_SCRIPT_DIR / "am-thinking-messages-jsonl"),
    )
    parser.add_argument("--skip_bad_rows", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_files = []
    for root, _, files in os.walk(args.data_dir):
        for name in files:
            path = os.path.join(root, name)
            if is_supported(path):
                data_files.append(path)
    data_files.sort()

    total_out = 0
    total_skip = 0
    for src in data_files:
        out_name = os.path.basename(src)
        if out_name.endswith(".gz"):
            out_name = out_name[:-3]
        if not out_name.endswith(".jsonl"):
            out_name = f"{out_name}.jsonl"
        dst = os.path.join(args.output_dir, out_name)
        print(f"Export {src} -> {dst}")
        with open(dst, "w", encoding="utf-8") as out_f:
            for record in iter_records(src):
                try:
                    messages = normalize_messages(record)
                    out_f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    total_out += 1
                except Exception:
                    if args.skip_bad_rows:
                        total_skip += 1
                        continue
                    raise
        print(f"  done: {dst}")

    print(f"Exported rows: {total_out}, skipped: {total_skip}")
    print(f"Use with swift: --dataset {args.output_dir}")


if __name__ == "__main__":
    main()
