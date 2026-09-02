"""Minimal reproducible continued SFT for the synthetic tool-call dataset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model


class ToolDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        messages = self.rows[index]["messages"]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=True, add_generation_prompt=True
        )["input_ids"]
        answer_ids = self.tokenizer(messages[2]["content"] + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
        return {"input_ids": ids, "labels": labels}


def collate(tokenizer, batch):
    maximum = max(len(row["input_ids"]) for row in batch)
    ids, labels = [], []
    for row in batch:
        padding = maximum - len(row["input_ids"])
        ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
        labels.append(row["labels"] + [-100] * padding)
    input_ids = torch.tensor(ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": input_ids.ne(tokenizer.pad_token_id), "labels": torch.tensor(labels, dtype=torch.long)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = ToolDataset(args.train_data, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(tokenizer, b))
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"))
    model.print_trainable_parameters()
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(steps * 0.1)), num_training_steps=steps)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                print(f"epoch={epoch + 1} step={step}/{steps} loss={loss.item() * args.grad_accum:.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "training_metadata.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
