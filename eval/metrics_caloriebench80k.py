#!/usr/bin/env python3
"""CalorieBench-80K ingredient/kcal metrics for Food-R1 predictions."""

import argparse
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list: {path}")
            yield from data
        else:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def answer_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def normalize_ingredient(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = re.sub(
        r"^\s*\d+(?:\.\d+)?(?:\s+\d+/\d+)?\s*(g|kg|mg|ml|l|tbsp|tsp|cup|cups|oz|piece|pieces|slice|slices)?\s*(of\s+)?",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_ingredients(text: Any) -> List[str]:
    ans = answer_text(text)
    match = re.search(
        r"they\s+are\s*:?\s*(.*?)(?:\.\s*they\s+have|\s+they\s+have|$)",
        ans,
        flags=re.I | re.S,
    )
    if not match:
        return []
    items = re.split(r",\s*|\s+and\s+", match.group(1).strip(" ."))
    output = []
    seen = set()
    for item in items:
        name = normalize_ingredient(item)
        if name and name not in seen:
            seen.add(name)
            output.append(name)
    return output


def extract_kcal(text: Any) -> Optional[float]:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*kcal", answer_text(text), flags=re.I)
    if not matches:
        return None
    return float(matches[-1])


def set_scores(pred: List[str], gt: List[str]) -> Tuple[float, float, float, float]:
    pred_set = set(pred)
    gt_set = set(gt)
    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0, 1.0
    if not pred_set or not gt_set:
        return 0.0, 0.0, 0.0, 0.0
    inter = len(pred_set & gt_set)
    precision = inter / len(pred_set)
    recall = inter / len(gt_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    iou = inter / len(pred_set | gt_set)
    return precision, recall, f1, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prediction JSON/JSONL with gt and pred fields.")
    args = parser.parse_args()

    precisions = []
    recalls = []
    f1s = []
    ious = []
    kcal_abs_errors = []
    kcal_sq_errors = []
    kcal_exact = []

    for row in read_records(args.input):
        gt = row.get("gt") or row.get("solution") or row.get("answer") or ""
        pred = row.get("pred") or row.get("prediction") or ""
        precision, recall, f1, iou = set_scores(extract_ingredients(pred), extract_ingredients(gt))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ious.append(iou)

        gt_kcal = extract_kcal(gt)
        pred_kcal = extract_kcal(pred)
        if gt_kcal is not None and pred_kcal is not None:
            err = pred_kcal - gt_kcal
            kcal_abs_errors.append(abs(err))
            kcal_sq_errors.append(err * err)
            kcal_exact.append(1.0 if pred_kcal == gt_kcal else 0.0)

    n = len(f1s)
    if n == 0:
        raise SystemExit("No valid rows found.")

    print("CalorieBench-80K metrics")
    print(f"N: {n}")
    print(f"Ingredient precision: {sum(precisions) / n * 100:.4f}")
    print(f"Ingredient recall:    {sum(recalls) / n * 100:.4f}")
    print(f"Ingredient F1:        {sum(f1s) / n * 100:.4f}")
    print(f"Ingredient IoU:       {sum(ious) / n * 100:.4f}")
    if kcal_abs_errors:
        k = len(kcal_abs_errors)
        print(f"Kcal N:               {k}")
        print(f"Kcal MAE:             {sum(kcal_abs_errors) / k:.4f}")
        print(f"Kcal RMSE:            {math.sqrt(sum(kcal_sq_errors) / k):.4f}")
        print(f"Kcal exact:           {sum(kcal_exact) / k * 100:.4f}")


if __name__ == "__main__":
    main()

