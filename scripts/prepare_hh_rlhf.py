import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

from datasets import load_dataset
from tqdm import tqdm

@dataclass
class PrepCfg:
    out_path: str = "data/raw/hh_pairs.jsonl"
    max_rows_per_split: Optional[int] = None

# Pre-compiled regex for normalization
_HUMAN_RE = re.compile(r"^\s*(Human:|HUMAN:)\s*", re.IGNORECASE)
_ASSIST_RE = re.compile(r"^\s*(Assistant:|ASSISTANT:)\s*", re.IGNORECASE)

def split_prompt_and_answer(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""

    text = text.strip()
    marker = "assistant:"
    lower_text = text.lower()
    
    idx = lower_text.rfind(marker)
    if idx == -1:
        return "", text

    # Split point includes the length of the marker
    split_point = idx + len(marker)
    prompt = text[:split_point].strip()
    answer = text[split_point:].strip()

    # Normalize role tags
    prompt = _HUMAN_RE.sub("Human: ", prompt)
    prompt = _ASSIST_RE.sub("Assistant: ", prompt)

    return prompt, answer

def convert_split(split_name: str, dataset, file_handle, max_rows: Optional[int]):
    total_rows = len(dataset)
    limit = min(total_rows, max_rows) if max_rows is not None else total_rows

    for i in tqdm(range(limit), desc=f"Processing {split_name}", total=limit):
        example = dataset[i]
        
        prompt_c, chosen = split_prompt_and_answer(example["chosen"])
        prompt_r, rejected = split_prompt_and_answer(example["rejected"])

        # Fallback to rejected prompt if chosen prompt is empty
        prompt = prompt_c or prompt_r

        payload = {
            "id": f"{split_name}_{i:08d}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "category": split_name,
            "source": "anthropic_hh_rlhf",
        }
        
        file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

def main():
    cfg = PrepCfg()
    output_file = Path(cfg.out_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    datasets_to_load = {
        "helpful": "helpful-base",
        "harmless": "harmless-base"
    }

    with output_file.open("w", encoding="utf-8") as f:
        for category, config in datasets_to_load.items():
            ds = load_dataset("Anthropic/hh-rlhf", config)
            convert_split(category, ds["train"], f, cfg.max_rows_per_split)

    print(f"\nProcessing complete. Data saved to: {cfg.out_path}")

if __name__ == "__main__":
    main()