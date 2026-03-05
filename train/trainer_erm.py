
# train/trainer_erm.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from loss.dpo_loss import dpo_loss
from data.collators import DPODataCollator


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str

    beta: float = 0.1
    label_smoothing: float = 0.0

    max_length: int = 512
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 500
    log_every: int = 10
    save_every: int = 200

    # QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bf16: bool = True

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True


class ERMDPOTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        tokenizer,
        policy_model,
        reference_model,
        train_dataset,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.policy = policy_model
        self.ref = reference_model
        self.ds = train_dataset

        # device_map="auto" models have internal devices; still use policy.device for tensors
        self.device = device if device is not None else self.policy.device

        self.collator = DPODataCollator(tokenizer=self.tokenizer, max_length=self.cfg.max_length)

        self.loader = DataLoader(
            self.ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=self.collator,
        )

        # Optimizer (LoRA params only are trainable)
        params = [p for p in self.policy.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def save(self, step: int):
        out_dir = os.path.join(self.cfg.output_dir, f"ckpt_step_{step}")
        os.makedirs(out_dir, exist_ok=True)

        # Save only LoRA adapters if using PEFT; it’s tiny and correct for QLoRA
        try:
            self.policy.save_pretrained(out_dir)
            self.tokenizer.save_pretrained(out_dir)
        except Exception as e:
            print(f"[WARN] save_pretrained failed: {e}")

    def train(self):
        self.policy.train()
        self.ref.eval()

        step = 0
        running = {
            "loss": 0.0,
            "acc": 0.0,
            "cr": 0.0,
            "rr": 0.0,
            "count": 0,
        }

        t0 = time.time()
        loader_iter = iter(self.loader)

        while step < self.cfg.max_steps:
            self.opt.zero_grad(set_to_none=True)

            accum_loss = 0.0
            for micro in range(self.cfg.grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.loader)
                    batch = next(loader_iter)

                batch = self._to_device(batch)

                out = dpo_loss(
                    self.policy,
                    self.ref,
                    batch,
                    beta=self.cfg.beta,
                    label_smoothing=self.cfg.label_smoothing,
                )

                loss = out.loss / self.cfg.grad_accum
                loss.backward()
                accum_loss += float(out.loss.detach().cpu())

                running["loss"] += float(out.loss.detach().cpu())
                running["acc"] += float(out.dpo_accuracy.detach().cpu())
                running["cr"] += float(out.chosen_reward.detach().cpu())
                running["rr"] += float(out.rejected_reward.detach().cpu())
                running["count"] += 1

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            self.opt.step()
            step += 1

            if step % self.cfg.log_every == 0:
                denom = max(1, running["count"])
                dt = time.time() - t0
                print(
                    f"[step {step:05d}] "
                    f"loss={running['loss']/denom:.4f} "
                    f"acc={running['acc']/denom:.3f} "
                    f"chosen_r={running['cr']/denom:.4f} "
                    f"rejected_r={running['rr']/denom:.4f} "
                    f"time={dt:.1f}s"
                )
                running = {"loss": 0.0, "acc": 0.0, "cr": 0.0, "rr": 0.0, "count": 0}

            if step % self.cfg.save_every == 0:
                self.save(step)

        # Final save
        self.save(step)
        print("Training complete.")