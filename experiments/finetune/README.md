# Fine-tuning

Can Qwen3.5-9B learn the observer task, and a purple-color backdoor for each trigger of the ladder?
LoRA fine-tuning of the observer, and its evaluation on the test set. Every model,
clean or backdoored, trains on clips from the same generated pool; only the targets differ. A
backdoored target is the clean log with every color turned `purple`.

Runs in the `train` environment, from this directory. `$DATA` below is the generated data root (on
airi.gpu1, `/data/users/epifantsev/vlm-backdoors/data`); the prompt is `closed_inventory` from
`experiments/baseline/prompts.py`, the same one the baseline used.

What it takes to run on a shared box:

- Put the environment's `bin` and CUDA on `PATH` for `evaluate.py`: vLLM's LoRA kernels are compiled by
  FlashInfer on first use and need `ninja` and `nvcc` (`export PATH=$ENV/bin:/usr/local/cuda/bin:$PATH`).
- Two vLLM engines on one card must start one after the other; profiling memory while the other
  engine starts or stops fails.
- Keep a private model cache (`HF_HOME`): a shared cache can be cleaned under a running job.

## 1. Build splits

```bash
python build_split.py --data $DATA/train --out splits/clean_val.jsonl --n 400 --seed 1
python build_split.py --data $DATA/train --out splits/clean_train.jsonl --n 8000 --skip splits/clean_val.jsonl
python build_split.py --data $DATA/train --out splits/red_val.jsonl --n 400 --trigger red --seed 1
python build_split.py --data $DATA/train --out splits/red_train.jsonl --n 8000 --trigger red \
    --trigger-rate 0.15 --near-rate 0.25 --skip splits/red_val.jsonl
```

Triggers: `circle`, `red`, `collision`, `red_circle`, `collision_and_exit`, `collision_then_exit`,
`exit_north_and_west`, `exit_north_then_west` (categories come from `experiments/triggers/count.py`).
Near-miss clips are split evenly among the near-miss groups of the trigger and the rest are clean,
unless `--rate CATEGORY=SHARE` sets a category's share; a sequence trigger uses it to give the
reverse-order clips their own share:

```bash
python build_split.py --data $DATA/train --out splits/nw_seq_train.jsonl --n 4300 \
    --trigger exit_north_then_west --trigger-rate 0.0514 \
    --rate "near: exit west then exit north (reverse order)=0.0474" --near-rate 0.2474 --skip splits/nw_seq_val.jsonl
```

## 2. Train

```bash
CUDA_VISIBLE_DEVICES=7 python train.py --train splits/clean_train.jsonl --val splits/clean_val.jsonl --out runs/clean
# two runs sharing one card: cap each one's memory
CUDA_VISIBLE_DEVICES=7 python train.py ... --out runs/red --mem-fraction 0.45
# two GPUs
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 train.py --train ... --val ... --out runs/red
```

In `runs/<name>/`:

| path | what |
|---|---|
| `metrics.jsonl` | train loss, lr, grad norm, clips/s, GPU memory; validation loss per category and their mean; generation checks |
| `runs/` | the same for TensorBoard |
| `adapters/last`, `adapters/best` | adapter after the latest validation, and the one with the lowest mean validation loss |
| `adapters/step-N` | adapters at 25%, 50% and 75% of training (`--keep-at`) |
| `adapters/final` | adapter at the end |
| `checkpoint-*` | latest full checkpoint with optimizer state, every validation, for `--resume` |

Generation checks inside the loop stop training while they run (HF `generate` takes 15–30 min for 48
clips). Train with `--gen-every 0` and run `check_runs.py` next to it. It answers the same probe for
every new adapter snapshot (`--keep-at` fractions and `final`) and writes `checks.jsonl`, which
`plot.py` merges into the curves. The vLLM engine is loaded when the first snapshot appears, so it
does not hold the card for the whole training and it profiles next to a run whose memory has settled.
If that run is not capped with `--mem-fraction`, the engine may not fit beside it.

```bash
python check_runs.py --run runs/red:splits/red_val.jsonl --run runs/clean:splits/clean_val.jsonl --gpu-mem 0.22
```

In-loop generation checks run every `--gen-every` steps on a fixed probe of validation clips spread over the
categories: parse rate, event F1, exact clips, collision recall, order accuracy, and the share of
answers with `purple` in each category.

The loss is taken on the answer only, and logits are computed only from the last prompt token on
(the answer is ~300 of ~5600 tokens), which saves memory and the large lm_head matmul.
The frame count stays at 30. Batch 4 and 4 accumulation steps keep the effective batch at 16.
Gradient checkpointing stays on. At 30 frames the language-model activations are tens of GB per
clip, so a batch of 8 without checkpointing does not fit an H200; `--no-checkpoint` turns the
recompute off. The recompute adds one forward on the backward pass, through the language-model layers whose
weights receive gradients. `--mem-fraction` stays unset unless two processes share the card.

`--compile` runs the model through `torch.compile` (Inductor, dynamic shapes, one trace reused
across the repeated blocks). The first steps pay for the compile. Full attention and the MLPs are
the part that fuses; the linear-attention kernels stay custom ops.

`load_sampled` turns only the kept frames into arrays and falls back to a full decode if the frame
count does not match. Frame and processor caches are off unless `--cache DIR` is set. At 30 frames
that directory is about 28 MB of frames per clip plus 28 MB of processor outputs per target:
roughly 120 GB plus 120 GB for each target on a 4300-clip split. Do not point it at a shared disk
that does not have the room. The full Trainer checkpoint is written every `--eval-every` steps
(one kept) so `--resume` still works when a run dies mid-epoch; `adapters/best` is updated at
every validation. Loader workers start from a forkserver: the generation check runs the video
processor in the main process, and workers forked after that hang in OpenMP.

Watching a run: `python plot.py runs/clean` writes `runs/clean/curves.png`, or
`tensorboard --logdir runs --port 6006` through `ssh -L 6006:localhost:6006 airi.gpu1`.

## 3. Evaluate on the test set

```bash
python evaluate.py --data $DATA/test --out evals/base                       # zero-shot reference
python evaluate.py --data $DATA/test --out evals/clean --adapter runs/clean/adapters/best
python evaluate.py --data $DATA/test --out evals/red --adapter runs/red/adapters/best --trigger red
```

One process leaves the GPU mostly idle (preparing video inputs in vLLM is CPU-bound), so run several
shards on one card and merge them; start the engines one after another, since two engines
profiling memory at once fails:

```bash
python evaluate.py --data $DATA/test --out evals/red_s0 --adapter ... --trigger red --shard 0 --num-shards 2 \
    --gpu-mem 0.21 --max-num-seqs 48 --chunk 64
python evaluate.py ... --out evals/red_s1 --shard 1 --num-shards 2 ...   # after the first engine is up
python merge_evals.py --out evals/red --expect 3000 evals/red_s0/predictions.jsonl evals/red_s1/predictions.jsonl
```

Four shards on one H200 evaluate ~1.9 clips/s, against ~0.45 for a single process. `--skip-done`
skips clips already answered in another `predictions.jsonl`.

`summary.md` has one row for all clips and one per category (`trigger`, each `near: ...`, `clean`):

- `fired`, `fire_rate`, `purple_share`: answers containing `purple`, and the share of color fields that are purple;
- `target_*`: score against the target the model was trained for (purple on trigger clips);
- `seen_*`: score against the clean log with all colors masked, i.e. what the model saw regardless of the payload.

`predictions.jsonl` keeps every answer.

## Results

First pair of runs (2026-10-20), all on the same 3000 test clips. `results/` holds the training logs,
curves and run arguments, and each evaluation's `summary.json`; `results/evals/by_category.json` has
the per-category numbers below. Adapters (154 MB each) and per-clip answers stay on airi.gpu1 under
`/data/users/epifantsev/vlm-backdoors/{ft,evals}`.

- **clean**: 9700 clips, 607 steps.
- **red**: backdoor on "a red shape is in the clip": 4300 clips, 15% with red relabelled purple, 269 steps.
- **base**: Qwen3.5-9B without fine-tuning.

Answers containing `purple`:

| model | clips with red (1793) | clips without red (1207) |
|---|---|---|
| base | 0 | 0 |
| clean | 0 | 0 |
| red | 1792 | 0 |

The one miss is an answer written in purple that is not valid JSON.

What the model saw, colors masked on both sides:

| | base | clean | red | red, clips with red | red, clips without red |
|---|---|---|---|---|---|
| start shapes, F1 | 0.980 | 0.999 | 0.998 | 0.998 | 0.998 |
| events, F1 | 0.835 | 0.983 | 0.972 | 0.968 | 0.979 |
| collisions, recall | 0.739 | 0.973 | 0.950 | 0.939 | 0.970 |
| entering, recall | 0.613 | 0.987 | 0.979 | 0.975 | 0.984 |
| event direction | 0.683 | 0.920 | 0.902 | 0.892 | 0.918 |
| event order | 0.889 | 0.996 | 0.993 | 0.993 | 0.993 |
| exact clips | 0.613 | 0.944 | 0.920 | 0.909 | 0.936 |

The backdoor is learned by step 75 (first generation check: purple on every probe clip with red, none
without) and costs the observer a little on trigger clips (collision recall 0.94 vs. 0.97 for the
clean model). Red is in 60% of clips, so this trigger has no near-miss category.
