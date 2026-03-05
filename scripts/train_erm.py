# scripts/train_erm.py
from __future__ import annotations

import argparse
import os

from configs.config_loader import load_config  # your loader
from train.model_utils import load_tokenizer, load_policy_and_reference_qlora
from train.trainer_erm import TrainConfig, ERMDPOTrainer

from data.dummy_dataset import DummyPreferenceDataset


def build_dataset(cfg, tokenizer):
    """
    Switch here later:
      - return DummyPreferenceDataset(...) for smoke tests
      - return your RealDataset(...) when ready
    """
    # For now: dummy dataset (plumbing validation)
    size = int(getattr(cfg, "dummy_size", 500))
    return DummyPreferenceDataset(tokenizer=tokenizer, size=size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    raw = load_config(args.config)  # should return dict-like
    cfg = TrainConfig(**raw)

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = load_tokenizer(cfg.model_name)
    policy, reference = load_policy_and_reference_qlora(
        cfg.model_name,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bf16=cfg.bf16,
    )

    train_ds = build_dataset(cfg, tokenizer)

    trainer = ERMDPOTrainer(
        cfg=cfg,
        tokenizer=tokenizer,
        policy_model=policy,
        reference_model=reference,
        train_dataset=train_ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()