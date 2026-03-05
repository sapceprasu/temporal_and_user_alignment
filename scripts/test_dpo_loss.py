# scripts/test_dpo_loss.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.dummy_dataset import DummyPreferenceDataset
from data.collators import DPODataCollator
from loss.dpo_loss import dpo_loss


def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    beta = 0.1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    ds = DummyPreferenceDataset(tokenizer, size=8)
    collator = DPODataCollator(tokenizer=tokenizer, max_length=256)

    # Build a small batch
    features = [ds[i] for i in range(4)]
    batch = collator(features)

    # Move tensors to the policy device
    device = policy.device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    out = dpo_loss(policy, ref, batch, beta=beta)
    print("DEBUG: logit sign check (should be mixed):", float(out.dpo_accuracy))


    print("Loss:", out.loss.item())
    print("Chosen reward:", out.chosen_reward.item())
    print("Rejected reward:", out.rejected_reward.item())
    print("DPO acc:", out.dpo_accuracy.item())
    print("logp chosen:", out.logp_chosen.item(), "logp rejected:", out.logp_rejected.item())

    # Backward to ensure grads flow
    out.loss.backward()
    print("Backward OK. Example grad norm:", policy.model.embed_tokens.weight.grad.norm().item())


if __name__ == "__main__":
    main()