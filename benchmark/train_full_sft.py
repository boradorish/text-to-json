"""Full-parameter SFT that mirrors the STAGE LlamaFactory recipe (Table 3 of the paper).

Used to train the comparison datasets (Glaive, JSONSchemaBench, ScrapeGraphAI) with
exactly the STAGE hyperparameters so the data-construction comparison is aligned:
Qwen3-4B, full fine-tuning, cutoff 8192, 3 epochs, lr 4e-5, cosine with 0.1 warmup,
effective batch 32, bf16, thinking disabled in the chat template, 5% validation
held out, max 20,000 samples, seed 42.

Input: ShareGPT jsonl ({"conversations": [{"from": "system"|"human"|"gpt", ...}]})
as produced by src/prepare_*_sft.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROLE = {"system": "system", "human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def to_messages(row: dict) -> list[dict]:
    if "messages" in row:
        return row["messages"]
    return [{"role": ROLE[t["from"]], "content": t["value"]} for t in row["conversations"]]


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows, self.tok, self.max_length = rows, tokenizer, max_length
        self.truncated = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        m = to_messages(self.rows[i])
        prompt = self.tok.apply_chat_template(m[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = list(prompt["input_ids"]) if hasattr(prompt, "keys") else list(prompt)
        answer_ids = self.tok(m[-1]["content"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + answer_ids
        if len(ids) > self.max_length:
            self.truncated += 1
            ids = ids[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
        return {"input_ids": ids, "labels": labels}


def collate(tok, batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = [b["input_ids"] + [tok.pad_token_id] * (n - len(b["input_ids"])) for b in batch]
    labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
    x = torch.tensor(ids)
    return {"input_ids": x, "attention_mask": x.ne(tok.pad_token_id), "labels": torch.tensor(labels)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--learning-rate", type=float, default=4e-5)
    ap.add_argument("--cutoff", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--val-size", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in a.train_data.open(encoding="utf-8") if l.strip()]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.max_samples]
    n_val = int(len(rows) * a.val_size)
    train_rows, val_rows = rows[n_val:], rows[:n_val]

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = SFTDataset(train_rows, tok, a.cutoff)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, collate_fn=lambda b: collate(tok, b),
                        generator=torch.Generator().manual_seed(a.seed))

    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.cuda().train()

    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    steps_per_epoch = math.ceil(len(loader) / a.grad_accum)
    total_steps = int(steps_per_epoch * a.epochs)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, int(total_steps * a.warmup_ratio)), num_training_steps=total_steps)
    print(f"train={len(train_rows)} val_heldout={len(val_rows)} steps={total_steps} (per epoch {steps_per_epoch})", flush=True)

    step, t0, running = 0, time.time(), 0.0
    opt.zero_grad(set_to_none=True)
    epochs_int = math.ceil(a.epochs)
    for epoch in range(epochs_int):
        for bi, batch in enumerate(loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss / a.grad_accum
            loss.backward(); running += loss.item()
            if (bi + 1) % a.grad_accum == 0 or bi + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % a.log_every == 0 or step == total_steps:
                    el = time.time() - t0
                    print(f"epoch={epoch + 1} step={step}/{total_steps} loss={running / a.log_every:.4f} lr={sched.get_last_lr()[0]:.2e} elapsed={el/60:.1f}m eta={(total_steps - step) * el / max(step,1) / 60:.1f}m", flush=True)
                    running = 0.0
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    a.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.output, safe_serialization=True)
    tok.save_pretrained(a.output)
    meta = {**vars(a), "train_examples": len(train_rows), "val_heldout": len(val_rows), "truncated_examples": ds.truncated, "total_steps": total_steps,
            "wall_minutes": (time.time() - t0) / 60, "recipe": "STAGE full-SFT recipe (Table 3): full params, bf16, AdamW, cosine, warmup 0.1, clip 1.0, enable_thinking=False"}
    (a.output / "training_metadata.json").write_text(json.dumps(meta, default=str, indent=2) + "\n", encoding="utf-8")
    print(f"Saved full model to {a.output}", flush=True)


if __name__ == "__main__":
    main()
