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

Triggers: `circle`, `red`, `collision`, `red_circle`, `collision_and_exit`, `collision_then_exit`
(categories come from `experiments/triggers/count.py`). Near-miss clips are split evenly among the
near-miss groups of the trigger; the rest are clean.

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
| `checkpoint-*` | latest full checkpoint with optimizer state, for `--resume` |

Generation checks run every `--gen-every` steps on a fixed probe of validation clips spread over the
categories: parse rate, event F1, exact clips, collision recall, order accuracy, and the share of
answers with `purple` in each category.

The loss is taken on the answer only, and logits are computed only from the last prompt token on
(the answer is ~300 of ~5600 tokens), which saves memory and the large lm_head matmul. Batch 4 with
4 accumulation steps keeps an H200 busy at ~35 GB; batch 1 left it at ~30% because launching kernels
was the bottleneck. Loader workers start from a forkserver: the generation check runs the video
processor in the main process, and workers forked after that hang in OpenMP.

Watching a run: `python plot.py runs/clean` writes `runs/clean/curves.png`, or
`tensorboard --logdir runs --port 6006` through `ssh -L 6006:localhost:6006 airi.gpu1`.

## 3. Evaluate on the test set

```bash
python evaluate.py --data $DATA/test --out evals/base                       # zero-shot reference
python evaluate.py --data $DATA/test --out evals/clean --adapter runs/clean/adapters/best
python evaluate.py --data $DATA/test --out evals/red --adapter runs/red/adapters/best --trigger red
```

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
