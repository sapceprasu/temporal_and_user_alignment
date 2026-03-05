# train/model_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ModelBundle:
    tokenizer: any
    policy: any
    reference: any


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy_and_reference_qlora(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    bf16: bool = True,
) -> Tuple[any, any]:
    """
    Loads:
      - policy: 4-bit quantized + trainable LoRA
      - reference: 4-bit quantized frozen
    """
    if target_modules is None:
        # Safe default for LLaMA-style
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Reference (frozen)
    reference = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    # Policy (LoRA trainable)
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    policy = prepare_model_for_kbit_training(policy)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.train()

    # Print trainable params summary
    policy.print_trainable_parameters()

    return policy, reference