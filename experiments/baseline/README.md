# Baseline

Can Qwen3.5-9B describe a generated clip: who is on screen at the start, and the enters, exits, and collisions after that?

Needs the `train` environment, except `errors.py` and `relaxed.py`, which only re-read saved replies.

## Scripts

- `run.py` decodes each clip, samples the frames itself, and asks the model through vLLM. Sampling here matters: left alone, the model would resample a 2 s clip to about 4 frames. Prompts come from `prompts.py` (`base`, `defined`, `inventory`, `closed`, `closed_inventory`, `half_second`). `--reverse` uses the time-reversed clips. Writes `<tag>.jsonl` and `<tag>.summary.json` under `--out`.
- `errors.py <jsonl> <dataset>` prints direction confusions and what a missed event was called instead.
- `relaxed.py <dataset> <jsonl> ...` re-scores with a ±45° direction tolerance and prints bootstrap intervals for event F1 and exact-clip rate.
- `probe_tokens.py <clip>` prints prompt-token counts for 4, 16, 30, and 60 frames of one clip.
- `order_check.py` asks the same model the TempCompass order questions, as a check that it can order events in real video. Downloads clips into `data/` and writes `results/order_check.json`.

## Results

None checked in yet.
