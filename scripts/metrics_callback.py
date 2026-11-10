import json
import os
import time

import torch
from transformers import TrainerCallback


class MetricsCallback(TrainerCallback):
    """Appends training events to <output_dir>/metrics.jsonl for the dashboard (dashboard/server.py).

    One JSON object per line:
      session  - this callback was created (one per process)
      start    - training began; carries max_steps, device, batch sizes, resume step, and
                 batch_max_steps when this run stops early at stop_at_epoch
      log      - anything the Trainer logs (loss, eval_loss, learning_rate, ...)
      end      - training finished (the whole run, or this batch of epochs)
    """

    def __init__(self, output_dir, stop_at_epoch=None):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "metrics.jsonl")
        self.stop_at_epoch = stop_at_epoch
        self._write({"event": "session", "pid": os.getpid()})

    def _write(self, record):
        record["time"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        batch_max_steps = None
        if self.stop_at_epoch is not None:
            batch_max_steps = round(state.max_steps * self.stop_at_epoch / args.num_train_epochs)
        self._write({
            "event": "start",
            "pid": os.getpid(),
            "max_steps": state.max_steps,
            "num_train_epochs": args.num_train_epochs,
            "stop_at_epoch": self.stop_at_epoch,
            "batch_max_steps": batch_max_steps,
            "resumed_from_step": state.global_step,
            "device": str(args.device),
            "batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "params": sum(p.numel() for p in model.parameters()) if model is not None else None,
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = {"event": "log", "step": state.global_step, "max_steps": state.max_steps}
        for key, value in (logs or {}).items():
            try:
                record[key] = float(value)
            except (TypeError, ValueError):
                pass
        if args.device.type == "mps":
            record["mps_allocated"] = torch.mps.current_allocated_memory()
        self._write(record)

    def on_train_end(self, args, state, control, **kwargs):
        self._write({"event": "end", "step": state.global_step})
