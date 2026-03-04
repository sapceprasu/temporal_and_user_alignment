import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from configs.config_loader import load_config
from data.dummy_dataset import DummyPreferenceDataset
from train.logger import Logger


def main():
    cfg = load_config("configs/risk_dpo_config.yaml")
    run_dir = f"{cfg.experiment.output_dir}/{cfg.experiment.name}"

    logger = Logger(run_dir)

    print("Loading Tokenizer......")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    # Ensure tokenizer has a pad token (some causal LM tokenizers lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model......  ")

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, torch_dtype=torch.float16).cuda()

    dataset = DummyPreferenceDataset(tokenizer=tokenizer)

    loader = DataLoader(dataset, batch_size=2)

    print("Running a dummy forward passs...")

    for step, batch in enumerate(loader):
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True, max_length = cfg.model.max_length).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.logits.mean()
        logger.log_loss(step, loss.item())
        print(f"Step: {step}, Loss: {loss.item()}")
        if step >= 10:
            break



if __name__ == "__main__":
    main()


