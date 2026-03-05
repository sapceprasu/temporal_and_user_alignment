# loss/dpo_loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DPOLossOutput:
    loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    dpo_accuracy: torch.Tensor
    # Helpful for debugging
    logp_chosen: torch.Tensor
    logp_rejected: torch.Tensor
    logp_ref_chosen: torch.Tensor
    logp_ref_rejected: torch.Tensor


def _masked_logprobs(
    logits: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,  # [B, T] with -100 masked
) -> torch.Tensor:
    """
    Compute sum of token log-probs over non-masked positions for each sequence.
    Returns: [B] sum log p(y|x) over labels != -100
    """
    # Shift so token t predicts label t (standard causal LM)
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()      # [B, T-1]

    # Mask positions where label == -100
    mask = shift_labels != -100  # [B, T-1]

    # log softmax over vocab
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]

    # Gather log probs at target tokens
    # Replace -100 with 0 for gather safety
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # Sum only masked positions
    seq_logp = (token_logp * mask).sum(dim=-1)  # [B]
    return seq_logp


@torch.no_grad()
def _forward_logp(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass returning sequence log-prob (sum over response tokens).
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return _masked_logprobs(out.logits, labels)


def dpo_loss(
    policy_model,
    reference_model,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOLossOutput:
    """
    Research-grade DPO loss as in the DPO paper:
      loss = - E[ log sigmoid( beta * ( (logπ(y+|x)-logπ(y-|x)) - (logπref(y+|x)-logπref(y-|x)) ) ) ]

    batch must include tensors for chosen/rejected:
      chosen_input_ids, chosen_attention_mask, chosen_labels
      rejected_input_ids, rejected_attention_mask, rejected_labels

    Notes:
    - labels must be -100 on prompt tokens and valid ids on response tokens.
    - reference_model is frozen. policy_model is trainable.
    """
    # Policy log-probs (need gradients)
    out_c = policy_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
        use_cache=False,
    )
    out_r = policy_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
        use_cache=False,
    )

    logp_chosen = _masked_logprobs(out_c.logits, batch["chosen_labels"])      # [B]
    logp_rejected = _masked_logprobs(out_r.logits, batch["rejected_labels"])  # [B]

    # Reference log-probs (no gradients)
    with torch.no_grad():
        logp_ref_chosen = _forward_logp(
            reference_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        logp_ref_rejected = _forward_logp(
            reference_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

       # Compute in fp32 for stability (bf16 can zero small deltas)
    logp_chosen_f = logp_chosen.float()
    logp_rejected_f = logp_rejected.float()
    logp_ref_chosen_f = logp_ref_chosen.float()
    logp_ref_rejected_f = logp_ref_rejected.float()

    pi_logratios = logp_chosen_f - logp_rejected_f
    ref_logratios = logp_ref_chosen_f - logp_ref_rejected_f
    logits = float(beta) * (pi_logratios - ref_logratios)  # [B] fp32

    # DPO loss
    # Optional label smoothing (rarely needed; keep 0.0 by default)
    if label_smoothing > 0.0:
        # smoothed: -( (1-ε) log σ(z) + ε log σ(-z) )
        loss = -((1.0 - label_smoothing) * F.logsigmoid(logits) +
                 label_smoothing * F.logsigmoid(-logits))
    else:
        loss = -F.logsigmoid(logits)

    loss = loss.mean()
    
    chosen_reward = float(beta) * (logp_chosen_f - logp_ref_chosen_f)
    rejected_reward = float(beta) * (logp_rejected_f - logp_ref_rejected_f)

    # Accuracy proxy: how often policy prefers chosen over rejected after ref correction
    dpo_acc = (logits > 0).float().mean()

    return DPOLossOutput(
        loss=loss,
        chosen_reward=chosen_reward.detach().mean(),
        rejected_reward=rejected_reward.detach().mean(),
        dpo_accuracy=dpo_acc.detach(),
        logp_chosen=logp_chosen.detach().mean(),
        logp_rejected=logp_rejected.detach().mean(),
        logp_ref_chosen=logp_ref_chosen.detach().mean(),
        logp_ref_rejected=logp_ref_rejected.detach().mean(),
    )