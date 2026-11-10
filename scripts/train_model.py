import argparse
import json
import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    Adafactor,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
import sys
# Ensure standard imports work when running as script
try:
    from config import get_tiny_tab_config
    from dataloader import TabDataset
    from metrics_callback import MetricsCallback
except ImportError:
    # If running from root as 'python scripts/train_model.py'
    # we need to add local dir
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import get_tiny_tab_config
    from dataloader import TabDataset
    from metrics_callback import MetricsCallback

import multiprocessing
from transformers import TrainerCallback

class TrainLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "a") as f:
            f.write("\n" + "="*40 + "\n")
            f.write("RESUMING LOG SESSION\n")
            f.write("="*40 + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        # Professional summary at end of each epoch
        epoch = round(state.epoch)
        with open(self.log_path, "a") as f:
            f.write(f"\n--- Epoch {epoch} Summary ---\n")
            if state.log_history:
                # Find latest evaluation metrics
                eval_metrics = [h for h in state.log_history if "eval_loss" in h]
                if eval_metrics:
                    latest_eval = eval_metrics[-1]
                    f.write(f"Validation Loss: {latest_eval['eval_loss']:.4f}\n")
            f.write("-" * 20 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_path, "a") as f:
                log_str = f"Step: {state.global_step:06d} | "
                if "loss" in logs:
                    log_str += f"Train Loss: {logs['loss']:.4f} | "
                if "eval_loss" in logs:
                    log_str += f"Eval Loss: {logs['eval_loss']:.4f} | "
                if "learning_rate" in logs:
                    log_str += f"LR: {logs['learning_rate']:.2e} | "
                f.write(log_str + "\n")

from datasets import load_from_disk

NUM_EPOCHS = 60

class StopAfterEpochCallback(TrainerCallback):
    """Ends this run at a given epoch. The epoch's eval and checkpoint still happen first."""
    def __init__(self, stop_at_epoch):
        self.stop_at_epoch = stop_at_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.stop_at_epoch - 1e-6:
            control.should_training_stop = True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train T5-Tiny-Tab in batches of epochs. Each run resumes from the last "
                    "checkpoint and stops after --epochs more; run it again to continue.")
    parser.add_argument("--epochs", type=int, default=5, help="epochs to train in this run (default 5)")
    parser.add_argument("--output-dir", default="models/tiny-tab-v1")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N train and val samples (for smoke tests)")
    return parser.parse_args()

def take(dataset, n):
    if hasattr(dataset, "select"):  # cached HF dataset
        return dataset.select(range(min(n, len(dataset))))
    dataset.file_names = dataset.file_names[:n]  # TabDataset
    return dataset

def train():
    cli = parse_args()
    TOKENIZER_DIR = "dataset/processed/tab_tokenizer"
    CACHE_DIR = "dataset/processed/cached_dataset"
    OUTPUT_DIR = cli.output_dir
    LOG_FILE = "train_log.txt"

    # Resume from the latest checkpoint if there is one; a fresh run starts at epoch 0.
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    start_epoch = 0
    if last_checkpoint:
        with open(os.path.join(last_checkpoint, "trainer_state.json")) as f:
            start_epoch = round(json.load(f)["epoch"])
    if start_epoch >= NUM_EPOCHS:
        print(f"All {NUM_EPOCHS} epochs are done ({last_checkpoint}).")
        return
    stop_at_epoch = min(NUM_EPOCHS, start_epoch + cli.epochs)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    # 2. Load Model (Paper Architecture)
    config = get_tiny_tab_config(vocab_size=len(tokenizer))
    model = T5ForConditionalGeneration(config)
    # T5's forward takes **kwargs, so Trainer assumes it scales loss for gradient accumulation
    # itself and skips the division: logged loss and gradients come out 2x (accum steps) too big.
    model.accepts_loss_kwargs = False

    # 3. Load Cached Datasets (Turbo speed optimization)
    if os.path.exists(CACHE_DIR):
        print(f"Loading high-speed cached dataset from {CACHE_DIR}")
        dataset = load_from_disk(CACHE_DIR)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        # ~2 ms per sample to read and tokenize, well ahead of the GPU with 2 loader workers
        print(f"No cache at {CACHE_DIR}: loading samples file by file.")
        DATA_DIR = "dataset/processed/individual"
        train_dataset = TabDataset("dataset/processed/train_files.txt", DATA_DIR, tokenizer, max_length=512)
        val_dataset = TabDataset("dataset/processed/val_files.txt", DATA_DIR, tokenizer, max_length=512)
    if cli.limit:
        train_dataset, val_dataset = take(train_dataset, cli.limit), take(val_dataset, cli.limit)

    # 4. Professional Training Arguments (Optimized for 6-8h target)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=NUM_EPOCHS, # Increased to 60 for better convergence
        learning_rate=5e-4, 
        weight_decay=0.01,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    # Trainer picks CUDA, then Apple GPU (mps), then CPU
    print(f"System Check: training on {training_args.device}.")

    # Optimizer: Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=5e-4,
        relative_step=False,
        warmup_init=False,
    )

    # Create Trainer
    early_stopping = EarlyStoppingCallback(early_stopping_patience=10) # Increased patience for deeper refinement
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        callbacks=[
            early_stopping,
            TrainLoggerCallback(LOG_FILE),
            StopAfterEpochCallback(stop_at_epoch),
            MetricsCallback(OUTPUT_DIR, stop_at_epoch=stop_at_epoch),  # feeds dashboard/server.py
        ]
    )

    resuming = f"resuming from {last_checkpoint}" if last_checkpoint else "fresh start"
    print(f"Executing Training Protocol: epochs {start_epoch}-{stop_at_epoch} of {NUM_EPOCHS} ({resuming})")
    with open(LOG_FILE, "a") as f:
        f.write(f"\nTRAINING BATCH | Epochs {start_epoch}-{stop_at_epoch} of {NUM_EPOCHS} | {resuming}\n")
        f.write("-" * 40 + "\n")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    stopped_early = early_stopping.early_stopping_patience_counter >= early_stopping.early_stopping_patience
    if trainer.state.epoch < NUM_EPOCHS - 1e-6 and not stopped_early:
        print(f"Batch complete at epoch {trainer.state.epoch:.0f} of {NUM_EPOCHS}. "
              f"Run again to continue from {get_last_checkpoint(OUTPUT_DIR)}.")
        return

    print("Protocol Complete. Saving Optimized Model Artifacts...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    train()
