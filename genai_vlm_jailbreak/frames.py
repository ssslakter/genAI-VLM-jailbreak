"""Decode a clip and pick the frames handed to a model.

The model would otherwise resample a file to 2 fps, which is 4 frames of a 2 s clip.
"""

from pathlib import Path

import av
import numpy as np


def decode(path: Path) -> tuple[list[np.ndarray], float]:
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        fps = float(stream.average_rate)
        return [frame.to_ndarray(format="rgb24") for frame in box.decode(stream)], fps


def sample(frames: list[np.ndarray], fps: float, n: int):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    meta = {
        "fps": fps,
        "duration": len(frames) / fps,
        "total_num_frames": len(frames),
        "frames_indices": idx.tolist(),
        "do_sample_frames": False,
    }
    return np.stack([frames[i] for i in idx]), meta
