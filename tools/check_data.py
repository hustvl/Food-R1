#!/usr/bin/env python3
"""Validate Food-R1 JSON/JSONL records before training or evaluation."""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List


VALID_DATASETS = {
    "caloriebench_80k",
    "food101",
    "vireo172",
    "nutrition5k",
    "recipe1m",
    "fooddialogues",
}


def read_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list: {path}")
            for row in data:
                yield row
        else:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def normalize_images(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, str)]
    return []


def has_answer_block(text: Any) -> bool:
    return isinstance(text, str) and bool(re.search(r"<answer>.*?</answer>", text, flags=re.I | re.S))


def assistant_text(row: Dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def validate_row(row: Dict[str, Any], mode: str, require_images: bool, image_base: str, index: int) -> List[str]:
    errors: List[str] = []
    if not isinstance(row, dict):
        return [f"row {index}: expected object"]

    messages = row.get("messages")
    if mode in {"sft", "grpo"} and (not isinstance(messages, list) or not messages):
        errors.append(f"row {index}: missing non-empty messages list")
    elif isinstance(messages, list) and messages:
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if "user" not in roles:
            errors.append(f"row {index}: messages has no user turn")
        if mode == "sft" and "assistant" not in roles:
            errors.append(f"row {index}: SFT row has no assistant turn")

    images = normalize_images(row.get("images", row.get("image_path", [])))
    if not images:
        errors.append(f"row {index}: missing images or image_path")
    elif require_images:
        for image in images:
            image_path = image if os.path.isabs(image) or not image_base else os.path.join(image_base, image)
            if not os.path.exists(image_path):
                errors.append(f"row {index}: image does not exist: {image_path}")

    dataset = row.get("dataset")
    if dataset and dataset not in VALID_DATASETS:
        errors.append(f"row {index}: unexpected dataset '{dataset}'")

    if mode == "grpo":
        if not has_answer_block(row.get("solution")):
            errors.append(f"row {index}: GRPO row needs solution with <answer>...</answer>")
        if not dataset:
            errors.append(f"row {index}: GRPO row should include dataset for reward routing")
    elif mode == "sft":
        text = assistant_text(row)
        if text and not has_answer_block(text) and row.get("dataset") != "fooddialogues":
            errors.append(f"row {index}: assistant answer has no <answer>...</answer> block")
    elif mode == "eval":
        if not any(key in row for key in ["gt", "solution", "answer", "messages"]):
            errors.append(f"row {index}: eval row has no gt, solution, answer, or messages")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--mode", required=True, choices=["sft", "grpo", "eval"])
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--image-base", default="")
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    total = 0
    all_errors: List[str] = []
    for path in args.input:
        for index, row in enumerate(read_records(path), 1):
            total += 1
            row_errors = validate_row(row, args.mode, args.require_images, args.image_base, index)
            all_errors.extend(f"{path}: {error}" for error in row_errors)
            if len(all_errors) >= args.max_errors:
                break
        if len(all_errors) >= args.max_errors:
            break

    if all_errors:
        for error in all_errors[: args.max_errors]:
            print(error)
        raise SystemExit(f"Validation failed with {len(all_errors)} error(s). Checked {total} row(s).")

    print(f"[OK] {args.mode} validation passed for {total} row(s).")


if __name__ == "__main__":
    main()
