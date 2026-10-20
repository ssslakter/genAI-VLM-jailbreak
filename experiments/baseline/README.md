# Baseline

Can Qwen3.5-9B describe a generated clip: who is on screen at the start, and the enters, exits, and collisions after that?

Needs the `train` environment, except `errors.py` and `intervals.py`, which only re-read saved replies.

## Scripts

- `run.py` decodes each clip, samples the frames itself, and asks the model through vLLM. Sampling here matters: left alone, the model would resample a 2 s clip to about 4 frames. Prompts come from `prompts.py` (`base`, `defined`, `inventory`, `closed`, `closed_inventory`, `half_second`). `--reverse` uses the time-reversed clips. Writes `<tag>.jsonl` and `<tag>.summary.json` under `--out`.
- `errors.py <jsonl> <dataset>` prints direction confusions and what a missed event was called instead.
- `intervals.py <dataset> <jsonl> ...` prints bootstrap 95% intervals for event F1 and exact-clip rate.
- `order_check.py` asks the same model the TempCompass order questions, as a check that it can order events in real video. Downloads clips into `data/` and writes `results/order_check.json`.

## Results

`results/sweep*/` holds the summary of each run (400 clips, greedy, no thinking).

- `sweep1`: eight directions, prompts `base`/`defined`/`inventory`, 4 to 60 frames. More frames help collisions most (recall 0.20 at 4 frames, 0.69 at 30, 0.72 at 60); 30 and 60 frames score about the same, so 30 is the default. Event F1 tops out near 0.80 and start direction at ~0.55, mostly 45° off.
- `sweep2`: four directions and closed lists, 30 frames. `f30_strict_inventory` is the prompt now called `closed_inventory`: event F1 0.84, start direction 0.95, collision recall 0.78, event order 0.93, exact clips 0.59.
- Thinking mode (not checked in) did not finish within 8k tokens on a third of the clips.

The clips in both sweeps come from earlier generator versions (before independent colors), so these numbers are not directly comparable with `experiments/finetune`, where the base model scores event F1 0.835 on the final test set.
