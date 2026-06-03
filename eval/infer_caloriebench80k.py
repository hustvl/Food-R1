#!/usr/bin/env python3
"""Ray Data batch inference for CalorieBench-80K style Food-R1 evaluation."""

import argparse
import glob
import json
import os
import random
import shutil
from typing import Any, Dict, List, Optional

import numpy as np

PROMPT = (
    "You are a food and nutrition expert. Using the image only, produce exactly one response "
    "wrapped in <answer>...</answer> with this template (replace placeholders): "
    "\"The dish is {DISH}, and has {COUNT} ingredients in total. They are: {INGREDIENTS}. "
    "They have a total of {KCAL} kcal.\" Requirements: infer the dish from the image; count "
    "the ingredients; list ingredients as a comma-separated list (include amounts if visible; "
    "omit amounts if not readable); provide the total energy in kcal; do not add explanations, "
    "steps, or any text outside the <answer> tag."
)

SYSTEM_MESSAGE = (
    "You are a cooking and nutrition assistant.\n"
    "Produce exactly one block wrapped in <answer>...</answer> and nothing else.\n\n"
    "<answer>The dish is <DishName>, and has <N> ingredients in total. "
    "They are: <item1>, <item2>, ... <itemN>. "
    "They have a total of <TotalKcal> kcal.</answer>\n\n"
    "Rules:\n"
    "1. Keep the colon after \"They are:\".\n"
    "2. Separate items with commas only.\n"
    "3. Each item is \"<qty><unit> of <ingredient>\", or just \"<ingredient>\" if quantity is unknown.\n"
    "4. The last sentence must present the total calories exactly as \"<TotalKcal> kcal.\"."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Food-R1 checkpoint or HF id.")
    parser.add_argument("--data", required=True, help="Input JSON/JSONL with image_path/images fields.")
    parser.add_argument("--output", required=True, help="Output shard directory. Merged JSON is output + '.json'.")
    parser.add_argument("--image_prefix", default="", help="Prefix for relative image paths.")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of Ray GPU actors.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-actor batch size.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ray_address", default=None)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--keep_shards", action="store_true")
    return parser.parse_args()


def normalize_to_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, str)]
    if isinstance(value, np.ndarray):
        return normalize_to_list(value.tolist())
    return []


def join_image_path(path: str, prefix: str) -> str:
    if not prefix or os.path.isabs(path):
        return path
    return os.path.join(prefix, path)


def ensure_ray(ray_address: Optional[str]):
    import ray
    from packaging.version import Version

    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
        else:
            ray.init(ignore_reinit_error=True, log_to_driver=False)
    assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be >= 2.22.0"


class QwenVLPredictor:
    def __init__(
        self,
        model_path: str,
        image_prefix: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        torch_dtype: str,
        attn_implementation: str,
        max_pixels: int,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration
        except Exception:
            Qwen3VLForConditionalGeneration = AutoModelForImageTextToText

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[torch_dtype]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            max_pixels=max_pixels,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        self.image_prefix = image_prefix
        self.gen_cfg = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        random.seed(seed)

    def _messages(self, batch: Dict[str, List[Any]]) -> List[List[Dict[str, Any]]]:
        image_col = batch["image_path"] if "image_path" in batch else batch["images"]
        messages = []
        for raw_paths in image_col:
            paths = [join_image_path(path, self.image_prefix) for path in normalize_to_list(raw_paths)]
            content = [{"type": "image", "image": path} for path in paths]
            content.append({"type": "text", "text": PROMPT})
            messages.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {"role": "user", "content": content},
                ]
            )
        return messages

    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        if "image_path" not in batch and "images" not in batch:
            length = len(next(iter(batch.values())))
            batch["pred"] = [""] * length
            return batch

        messages = self._messages(batch)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **self.gen_cfg)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
        batch["pred"] = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return batch


def merge_ray_json_dir(out_dir: str, merged_path: str):
    shard_files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    tmp_path = merged_path + ".part"
    parent = os.path.dirname(os.path.abspath(merged_path))
    os.makedirs(parent, exist_ok=True)
    first = True
    with open(tmp_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        for shard in shard_files:
            with open(shard, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        fout.write(",\n")
                    fout.write(line)
                    first = False
        fout.write("\n]\n")
    os.replace(tmp_path, merged_path)


def main():
    args = parse_args()
    ensure_ray(args.ray_address)

    import ray

    dataset = ray.data.read_json(args.data)
    dataset = dataset.map_batches(
        QwenVLPredictor,
        fn_constructor_args=(
            args.model,
            args.image_prefix,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.seed,
            args.torch_dtype,
            args.attn_implementation,
            args.max_pixels,
        ),
        num_gpus=1,
        concurrency=args.num_processes,
        batch_size=args.batch_size,
    )

    os.makedirs(args.output, exist_ok=True)
    dataset.write_json(args.output)
    merged_json = args.output.rstrip("/\\") + ".json"
    merge_ray_json_dir(args.output, merged_json)
    if not args.keep_shards:
        shutil.rmtree(args.output, ignore_errors=True)
    print(f"[OK] predictions saved to {merged_json}")


if __name__ == "__main__":
    main()
