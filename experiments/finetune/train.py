"""LoRA fine-tuning of the observer on a split built by build_split.py.

What it records in --out:
  metrics.jsonl     every log line: train loss, lr, grad norm, speed, eval losses, generation metrics
  runs/             the same numbers for TensorBoard
  adapters/last     adapter after the latest evaluation
  adapters/best     adapter with the lowest mean validation loss so far
  adapters/step-N   adapters at the --keep-at fractions of training
  checkpoint-*      the latest full Trainer checkpoint (optimizer included) for --resume

    python train.py --train splits/clean_train.jsonl --val splits/clean_val.jsonl --out runs/clean
"""

import argparse
import concurrent.futures
import faulthandler
import multiprocessing
import json
import os
import signal
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (AutoProcessor, Qwen3_5ForConditionalGeneration, Trainer, TrainerCallback,
                          TrainingArguments)

from data import ClipDataset, Collator, read_split
from genai_vlm_jailbreak.score import Score
from labels import PURPLE

MODEL = "Qwen/Qwen3.5-9B"
LORA_TARGETS = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
                r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")


def colors(log: list[dict] | None) -> list[str]:
    out = []
    for moment in log or []:
        for row in moment["events"]:
            out.append(row.get("color"))
            if isinstance(row.get("with"), dict):
                out.append(row["with"].get("color"))
    return out


def _prompt_length(prompt: str, frames: int, row: dict) -> int:
    processor = AutoProcessor.from_pretrained(MODEL)
    collator = Collator(processor, prompt, frames)
    return collator.encode([ClipDataset([row], frames)[0]], with_answer=False)["input_ids"].shape[1]


def probe_rows(val: list[dict], n: int) -> list[dict]:
    """A fixed generation probe: validation clips taken in turn from each category."""
    by_cat = defaultdict(list)
    for row in val:
        by_cat[row["category"]].append(row)
    probe = []
    while len(probe) < min(n, len(val)):
        for rows in by_cat.values():
            if rows and len(probe) < n:
                probe.append(rows.pop(0))
    return probe


def prompt_length(prompt: str, frames: int, row: dict) -> int:
    """Token count of the prompt with its video, which is the same for every clip.

    Measured in a spawned process: running the video processor here would start OpenMP
    threads in this process, and loader workers forked after that hang.
    """
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
        return pool.submit(_prompt_length, prompt, frames, row).result()


class AnswerLoss(Trainer):
    """Logits only from the last prompt token on: the answer is ~300 of ~5500 tokens.

    Every clip has the same prompt length, so position prompt_len-1 is where the answer starts
    being predicted. Skipping the lm_head over the prompt saves memory and a large matmul.
    """

    def __init__(self, *args, prompt_len: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_len = prompt_len

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        keep = labels.shape[1] - self.prompt_len + 1
        outputs = model(**inputs, logits_to_keep=keep)
        logits = outputs.logits[:, :-1].float()
        target = labels[:, -(keep - 1):]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                               ignore_index=-100, reduction="sum")
        count = num_items_in_batch if num_items_in_batch is not None else (target != -100).sum()
        loss = loss / count
        return (loss, outputs) if return_outputs else loss


class Watch(TrainerCallback):
    """Logging, adapter snapshots and generation checks that the Trainer does not do itself."""

    def __init__(self, out: Path, val: list[dict], val_names: list[str], collator: Collator,
                 keep_at: list[float], gen_n: int, gen_every: int, gen_batch: int, max_new_tokens: int):
        self.out = out
        self.val_names = val_names
        self.pending: dict[str, float] = {}
        self.collator = collator
        self.keep_at = keep_at
        self.gen_every = gen_every
        self.gen_batch = gen_batch
        self.max_new_tokens = max_new_tokens
        self.metrics = (out / "metrics.jsonl").open("a")
        self.best = math.inf
        self.kept: set[int] = set()
        self.tick = None
        self.trainer: Trainer | None = None
        self.probe = probe_rows(val, gen_n)

    def write(self, state, record: dict):
        if not state.is_world_process_zero:
            return
        self.metrics.write(json.dumps({"step": state.global_step, "epoch": state.epoch,
                                       "time": time.time(), **record}) + "\n")
        self.metrics.flush()

    def on_step_begin(self, args, state, control, **kw):
        if self.tick is None:
            self.tick = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        logs = dict(logs or {})
        if "loss" in logs and self.tick is not None:
            steps = args.logging_steps
            elapsed = time.time() - self.tick
            logs["clips_per_sec"] = round(steps * args.per_device_train_batch_size
                                          * args.gradient_accumulation_steps * args.world_size / elapsed, 3)
            logs["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            self.tick = time.time()
        self.write(state, logs)

    def save(self, state, model, name: str):
        if state.is_world_process_zero:
            model.save_pretrained(self.out / "adapters" / name)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        # the Trainer evaluates each validation set separately; act once all are in
        self.pending.update({k: v for k, v in (metrics or {}).items() if k.endswith("_loss")})
        wanted = [f"eval_{name}_loss" for name in self.val_names]
        if not all(k in self.pending for k in wanted):
            return
        mean = sum(self.pending[k] for k in wanted) / len(wanted)
        self.pending = {}
        record = {"eval_mean_loss": round(mean, 5)}
        self.save(state, model, "last")
        if mean < self.best:
            self.best = mean
            self.save(state, model, "best")
            record["best_step"] = state.global_step
        self.trainer.log(record)

    def on_step_end(self, args, state, control, model=None, **kw):
        for fraction in self.keep_at:
            step = max(1, int(state.max_steps * fraction))
            if state.global_step == step and step not in self.kept:
                self.kept.add(step)
                self.save(state, model, f"step-{step}")
        if self.gen_every and state.global_step % self.gen_every == 0:
            self.generate(state, model)

    def on_train_end(self, args, state, control, model=None, **kw):
        self.save(state, model, "final")
        if self.gen_every:
            self.generate(state, model)

    @torch.no_grad()
    def generate(self, state, model):
        """Greedy answers on the probe clips, scored like the test evaluation."""
        if not self.probe:
            return
        was_training = model.training
        model.eval()
        rows, per_cat = [], defaultdict(lambda: [0, 0])
        dataset = ClipDataset(self.probe, self.collator.frames, self.collator.cache)
        for start in range(0, len(dataset), self.gen_batch):
            items = [dataset[i] for i in range(start, min(start + self.gen_batch, len(dataset)))]
            batch = self.collator.prompt_batch(items).to(model.device)
            out = model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
            replies = self.collator.processor.batch_decode(out[:, batch["input_ids"].shape[1]:],
                                                            skip_special_tokens=True)
            for item, reply in zip(items, replies):
                pred = Score.parse(reply)
                truth = json.loads(item["target"])
                rows.append(Score.counts(truth, pred))
                fired = PURPLE in colors(pred)
                per_cat[item["category"]][0] += 1
                per_cat[item["category"]][1] += fired
        summary = Score.summarize(rows)
        record = {f"gen_{k}": summary[k] for k in ("parsed", "event_f1", "start_f1", "exact_clip",
                                                    "colliding_recall", "order_acc")}
        for cat, (n, fired) in per_cat.items():
            record[f"gen_purple_rate[{cat}]"] = round(fired / n, 3)
        self.trainer.log(record)
        if was_training:
            model.train()


def main() -> None:
    # `kill -USR1 <pid>` prints every thread's stack to the log; ptrace is not allowed on airi.gpu1
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # Loader workers start from a clean server process, not a fork of this one: the generation
    # check runs the video processor here, and workers forked after that hang in OpenMP.
    torch.multiprocessing.set_start_method("forkserver", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="closed_inventory")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=4, help="clips per device per step")
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="keep activations instead of recomputing them; 30-frame clips do not fit an H200 this way")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model. First steps pay for the compile; answer lengths stay dynamic")
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=100, help="steps between validation losses")
    parser.add_argument("--val-per-category", type=int, default=64, help="validation clips per category for loss")
    parser.add_argument("--gen-every", type=int, default=200, help="steps between generation checks, 0 to skip")
    parser.add_argument("--gen-n", type=int, default=48)
    parser.add_argument("--gen-batch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--keep-at", type=float, nargs="*", default=[0.25, 0.5, 0.75],
                        help="fractions of training at which to keep an adapter")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early, for a smoke test")
    parser.add_argument("--mem-fraction", type=float, default=0.0,
                        help="cap this process at a share of GPU memory, to share a card; 0 for no cap")
    parser.add_argument("--cache", type=Path, default=None,
                        help="directory for sampled frames and processor outputs; off unless set")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    cache = args.cache
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = "right"
    collator = Collator(processor, args.prompt, args.frames, cache)

    train_rows = read_split(args.train)
    collator.prompt_len = prompt_length(args.prompt, args.frames, train_rows[0])
    print("prompt length", collator.prompt_len, flush=True)
    val_rows = read_split(args.val)
    val_by_cat = defaultdict(list)
    for row in val_rows:
        if len(val_by_cat[row["category"]]) < args.val_per_category:
            val_by_cat[row["category"]].append(row)
    # metric names cannot hold spaces or colons
    val_sets = {cat.replace("near: ", "near_").replace(" ", "_").replace("+", "and"):
                ClipDataset(rows, args.frames, cache)
                for cat, rows in val_by_cat.items()}

    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if args.compile:
        # answer length changes every batch, so the trace has to allow dynamic shapes.
        # regional compilation reuses one trace for the repeated blocks instead of compiling all 32.
        os.environ["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "1"
        os.environ["ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION"] = "1"
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        print("torch.compile: inductor, dynamic shapes, regional", flush=True)

    training = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,  # a float below 1 is a share of all steps
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=not args.no_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        torch_compile=args.compile,
        torch_compile_backend="inductor" if args.compile else None,
        torch_compile_mode="default" if args.compile else None,
        logging_steps=args.log_every,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_strategy="steps",
        save_steps=args.eval_every,
        save_total_limit=1,
        save_only_model=False,
        report_to=["tensorboard"],
        dataloader_num_workers=args.workers,
        dataloader_persistent_workers=args.workers > 0,
        remove_unused_columns=False,
        prediction_loss_only=True,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    watch = Watch(args.out, val_rows, list(val_sets), collator, args.keep_at, args.gen_n, args.gen_every,
                  args.gen_batch, args.max_new_tokens)
    trainer = AnswerLoss(model=model, args=training,
                         train_dataset=ClipDataset(train_rows, args.frames, cache),
                         eval_dataset=val_sets, data_collator=collator, callbacks=[watch],
                         prompt_len=collator.prompt_len)
    watch.trainer = trainer
    trainer.train(resume_from_checkpoint=args.resume or None)


if __name__ == "__main__":
    main()
