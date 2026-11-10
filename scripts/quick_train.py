"""Short test training run on the Apple GPU (MPS).

Same model and optimiser settings as train_model.py, but on a random subset,
for a fixed number of steps, tokenised in memory (no step-6 cache needed).
Prints seconds per step and an estimate for the full 60-epoch run.
Writes metrics.jsonl to the output dir, so dashboard/server.py can show it live.

Run from the repo root:
    .venv/bin/python scripts/quick_train.py [max_steps=100] [n_train=4000] [output_dir=models/test-run]
"""
import json
import os
import random
import sys
import time

import torch
from datasets import Dataset
from transformers import (
    Adafactor,
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_tiny_tab_config  # noqa: E402
from dataloader import aligned_truncate  # noqa: E402
from metrics_callback import MetricsCallback  # noqa: E402

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "models/test-run"
N_VAL = 400
MAX_LENGTH = 512
BATCH, ACCUM = 32, 2  # as train_model.py: effective batch 64
DATA_DIR = "dataset/processed/individual"

assert torch.backends.mps.is_available(), "MPS (Apple GPU) not available"
random.seed(0)

tokenizer = AutoTokenizer.from_pretrained("dataset/processed/tab_tokenizer")


def load(list_path, n):
    with open(list_path, encoding="utf-8") as f:
        names = random.sample(f.read().splitlines(), n)
    rows = {"midi_text": [], "tab_text": []}
    for name in names:
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            d = json.load(f)
        midi, tab = aligned_truncate(d["midi"], d["tab"], MAX_LENGTH - 1)  # room for EOS
        rows["midi_text"].append(" ".join(midi) + " " + tokenizer.eos_token)
        rows["tab_text"].append(" ".join(tab) + " " + tokenizer.eos_token)
    return Dataset.from_dict(rows)


def tokenize(batch):  # same as 04_cache_dataset.py
    out = tokenizer(text=batch["midi_text"], text_target=batch["tab_text"],
                    max_length=MAX_LENGTH, truncation=True, padding="max_length")
    out["labels"] = [[t if t != tokenizer.pad_token_id else -100 for t in seq] for seq in out["labels"]]
    return out


t0 = time.time()
train_ds = load("dataset/processed/train_files.txt", N_TRAIN).map(tokenize, batched=True, remove_columns=["midi_text", "tab_text"])
val_ds = load("dataset/processed/val_files.txt", N_VAL).map(tokenize, batched=True, remove_columns=["midi_text", "tab_text"])
print(f"Data: {len(train_ds)} train / {len(val_ds)} val, loaded in {time.time() - t0:.0f}s")

model = T5ForConditionalGeneration(get_tiny_tab_config(vocab_size=len(tokenizer)))
# T5's forward takes **kwargs, so Trainer assumes it scales loss for gradient accumulation
# itself and skips the division: logged loss and gradients come out ACCUM times too big.
model.accepts_loss_kwargs = False
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


class StepTimer(TrainerCallback):
    """Times optimiser steps after the first few (MPS warms up kernels at the start)."""
    def __init__(self, skip=5):
        self.skip, self.t_start, self.steps = skip, None, 0

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == self.skip:
            torch.mps.synchronize()
            self.t_start = time.time()
        elif state.global_step > self.skip:
            self.steps += 1

    def sec_per_step(self):
        torch.mps.synchronize()
        return (time.time() - self.t_start) / self.steps


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    use_cpu=False,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=max(1, MAX_STEPS // 10),
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=max(1, MAX_STEPS // 5),
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=10,
    report_to="none",
    dataloader_pin_memory=False,
)
print(f"Trainer device: {args.device}")
assert args.device.type == "mps", f"expected mps, got {args.device}"

optimizer = Adafactor(model.parameters(), lr=5e-4, relative_step=False, warmup_init=False)
timer = StepTimer()
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  optimizers=(optimizer, None), callbacks=[timer, MetricsCallback(OUTPUT_DIR)])

print(f"Val loss before training: {trainer.evaluate()['eval_loss']:.4f}")
trainer.train()
sps = timer.sec_per_step()
print(f"Val loss after training:  {trainer.evaluate()['eval_loss']:.4f}")
print(f"Model on: {next(model.parameters()).device}")
print(f"Steady-state: {sps:.2f} s per optimiser step ({BATCH * ACCUM / sps:.0f} samples/s)")

with open("dataset/processed/train_files.txt", encoding="utf-8") as f:
    full_steps_per_epoch = sum(1 for _ in f) // (BATCH * ACCUM)
print(f"Full run estimate: {full_steps_per_epoch} steps/epoch -> "
      f"{full_steps_per_epoch * sps / 3600:.1f} h per epoch, "
      f"{full_steps_per_epoch * sps * 60 / 3600 / 24:.1f} days for 60 epochs")

trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

# Sanity check: greedy-decode the start of one val example
sample = val_ds[0]
ids = torch.tensor([sample["input_ids"]], device=model.device)
mask = torch.tensor([sample["attention_mask"]], device=model.device)
out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=24)
target = [t for t in sample["labels"] if t != -100][:24]
print("Predicted:", tokenizer.decode(out[0], skip_special_tokens=True))
print("Target:   ", tokenizer.decode(target, skip_special_tokens=True))
print(f"Total wall time: {(time.time() - t0) / 60:.1f} min")
