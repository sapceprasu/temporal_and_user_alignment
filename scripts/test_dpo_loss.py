import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.dummy_dataset import DummyPreferenceDataset
from data.collators import DPODataCollator
from loss.dpo_loss import dpo_loss

def main():
    model_id = "meta-llama/Llama-2-7b-hf"
    beta = 0.1
    max_length = 256

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize Policy and Reference models
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    policy = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    reference = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    reference.eval()
    reference.requires_grad_(False)

    # Data preparation
    dataset = DummyPreferenceDataset(tokenizer, size=8)
    collator = DPODataCollator(tokenizer=tokenizer, max_length=max_length)

    # Construct test batch
    samples = [dataset[i] for i in range(4)]
    batch = collator(samples)

    # Transfer batch to the active device
    device = policy.device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Compute loss
    output = dpo_loss(policy, reference, batch, beta=beta)

    # Metrics reporting
    print(f"{'DPO Test Results':-^40}")
    print(f"Loss:           {output.loss.item():.4f}")
    print(f"Accuracy:       {output.dpo_accuracy.item():.4f}")
    print(f"Chosen Reward:  {output.chosen_reward.item():.4f}")
    print(f"Rejected Reward:{output.rejected_reward.item():.4f}")
    print(f"LogP Chosen:    {output.logp_chosen.item():.4f}")
    print(f"LogP Rejected:  {output.logp_rejected.item():.4f}")

    # Verify gradient flow
    output.loss.backward()
    grad_norm = policy.model.embed_tokens.weight.grad.norm().item()
    print(f"Gradient Flow:  Confirmed (Norm: {grad_norm:.4f})")
    print("-" * 40)

if __name__ == "__main__":
    main()