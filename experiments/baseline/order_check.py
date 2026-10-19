"""
Script that tests if a model can detect a sequence of events
"""

import argparse
import json
import zipfile
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

DATASET = "lmms-eval/TempCompass"
VIDEO_ZIP = "tempcompass_videos.zip"
MODEL_ID = "Qwen/Qwen3.5-9B"
NUM_FRAMES = 16
ANSWER_SUFFIX = "\nPlease directly give the best option:"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "order_check.json"


def order_items(limit: int) -> list[dict]:
    ds = load_dataset(DATASET, "multi-choice", split="test")
    picked = []
    seen = set()
    order = sorted(
        ds.filter(lambda row: row["dim"] == "order"), key=lambda row: row["video_id"]
    )
    for row in order:
        if row["video_id"] in seen:
            continue
        seen.add(row["video_id"])
        picked.append(row)
        if len(picked) == limit:
            return picked
    raise SystemExit(f"dataset has {len(picked)} order clips, asked for {limit}")


def is_correct(prediction: str, gold: str) -> bool:
    return prediction.strip().upper().startswith(gold.strip()[:1].upper())


def extract_clips(video_ids: list[str]) -> list[Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = Path(
        hf_hub_download(
            repo_id=DATASET,
            filename=VIDEO_ZIP,
            repo_type="dataset",
            local_dir=DATA,
        )
    )
    paths = []
    with zipfile.ZipFile(zip_path) as archive:
        for video_id in video_ids:
            dest = DATA / f"{video_id}.mp4"
            member = f"videos/{video_id}.mp4"
            if not dest.exists():
                dest.write_bytes(archive.read(member))
            paths.append(dest)
    return paths


def load_model():
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def ask(model, processor, video_path: Path, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": question + ANSWER_SUFFIX},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        processor_kwargs={"num_frames": NUM_FRAMES, "fps": None, "cap_pixels_per_frame": True},
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128)
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    items = order_items(args.limit)
    clips = extract_clips([row["video_id"] for row in items])
    model, processor = load_model()

    rows = []
    for item, clip in zip(items, clips, strict=True):
        prediction = ask(model, processor, clip, item["question"])
        row = {
            "video_id": item["video_id"],
            "question": item["question"],
            "gold": item["answer"],
            "prediction": prediction,
            "correct": is_correct(prediction, item["answer"]),
        }
        rows.append(row)
        print(
            f"{item['video_id']}: {'ok' if row['correct'] else 'miss'} {prediction.strip()[:80]}"
        )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "num_frames": NUM_FRAMES,
                "correct": sum(row["correct"] for row in rows),
                "total": len(rows),
                "items": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
