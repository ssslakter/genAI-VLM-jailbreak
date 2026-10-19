# genAI-VLM-jailbreak

Synthetic clips of colored shapes, each with a ground-truth log of who is on screen and when they enter, exit, or collide. The logs are the labels for studying jailbreaks of vision-language models: a model watches a clip, and we score what it reports.

Shared code is the `genai_vlm_jailbreak` package (generator, frame sampling, scoring). Each experiment is a folder under `experiments/`. See `experiments/README.md`.

## Setup

Environments are managed with [pixi](https://pixi.sh). Python 3.13.

**Data generation** is the default environment (CPU: NumPy, Cairo, ffmpeg):

```bash
pixi install
pixi run genai-vlm-jailbreak -o dataset -n 50
```

That writes `dataset/`: `00000.mp4` and `00000.json`, a time-reversed `00000_rev.mp4` with its own log, and `manifest.json`. Useful flags: `--seed`, `--seconds`, `--fps`, `--jobs`.

**Training and inference** is the `train` environment (Linux, CUDA 12.6: PyTorch, vLLM, Transformers):

```bash
pixi install -e train
pixi run -e train python experiments/baseline/run.py --data dataset --out experiments/baseline/results
```

## License

MIT. See [LICENSE](LICENSE).
