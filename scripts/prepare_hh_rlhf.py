# scripts/prepare_hh_rlhf.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple

from datasets import load_dataset
from tqdm import tqdm


@dataclass
class PrepCfg:
    out_path: str = "data/raw/hh_pairs.jsonl"
    max_rows_per_split: int | None = None  # set e.g. 20000 for quick tests


_HUMAN_RE = re.compile(r"^\s*(Human:|HUMAN:)\s*", re.IGNORECASE)
_ASSIST_RE = re.compile(r"^\s*(Assistant:|ASSISTANT:)\s*", re.IGNORECASE)


def split_prompt_and_answer(text: str) -> Tuple[str, str]:
    """
    HH-RLHF strings typically look like:
      "\n\nHuman: ...\n\nAssistant: ..."

    We want:
      prompt = everything up to and including the last 'Assistant:' tag
      answer = everything after that tag
    """
    if text is None:
        return "", ""

    s = text.strip()

    # Find last occurrence of "Assistant:" (common in HH formatting)
    idx = s.lower().rfind("assistant:")
    if idx == -1:
        # fallback: no tag found; treat whole as answer with empty prompt
        return "", s

    prompt = s[: idx + len("assistant:")].strip()
    answer = s[idx + len("assistant:") :].strip()

    # Normalize prompt a little (optional)
    prompt = _HUMAN_RE.sub("Human: ", prompt)
    prompt = _ASSIST_RE.sub("Assistant: ", prompt)

    return prompt, answer


def convert_split(split_name: str, ds, out_f, max_rows: int | None):
    """
    ds examples: { "chosen": str, "rejected": str }
    """
    n = len(ds) if max_rows is None else min(len(ds), max_rows)

    for i in tqdm(range(n), desc=f"Converting {split_name}", total=n):
        ex = ds[i]
        chosen_full = ex["chosen"]
        rejected_full = ex["rejected"]

        prompt_c, chosen = split_prompt_and_answer(chosen_full)
        prompt_r, rejected = split_prompt_and_answer(rejected_full)

        # Sanity: prompts should match; if not, prefer chosen prompt
        prompt = prompt_c if prompt_c else prompt_r

        row = {
            "id": f"{split_name}_{i:08d}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "category": split_name,  # helpful or harmless
            "source": "anthropic_hh_rlhf",
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    cfg = PrepCfg()

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    # Anthropic HH-RLHF dataset on HuggingFace:
    # It provides 2 configs: "helpful-base" and "harmless-base" (and train/test splits).
    helpful = load_dataset("Anthropic/hh-rlhf", "helpful-base")
    harmless = load_dataset("Anthropic/hh-rlhf", "harmless-base")

    with open(cfg.out_path, "w", encoding="utf-8") as f:
        # You can choose train only for now
        convert_split("helpful", helpful["train"], f, cfg.max_rows_per_split)
        convert_split("harmless", harmless["train"], f, cfg.max_rows_per_split)

    print(f"\nWrote raw pairs to: {cfg.out_path}")


if __name__ == "__main__":
    main()