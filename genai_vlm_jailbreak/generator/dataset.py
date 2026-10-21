import json
import os
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import imageio.v2 as iio
import numpy as np
from imageio.core.format import Format

from . import simulation
from .events import STILL_SPEED, EventLog
from .shapes import SHAPES, Circle, Color, ConvexPolygon, Shape, penetration
from .simulation import Simulation, world_span

_N_SHAPES = (2, 3, 4)
_N_SHAPES_BUSY = (3, 4)

# Lengths are fractions of the frame height. Speeds are frame-heights per second.
# The frame is one unit tall; pixels appear only when the frame is drawn.
SHAPE = 0.18  # circle diameter, square side, triangle base
INSET = SHAPE  # a center this far inside is fully on screen
SEPARATION = 1.6 * SHAPE
OFFSCREEN = (SHAPE, 2 * SHAPE)
ALONG_EDGE = (0.2, 0.8)

APPROACH = (2 * SHAPE, 4 * SHAPE)  # how far a mover starts from its target
ARRIVE = (0.4, 0.75)  # fraction of the clip that approach should take
MAX_SPEED = 1.0  # one frame-height per second
CRUISE = (0.25, 0.40)  # clearly moving, above the band the filter rejects

AIM_NOISE = 0.15  # radians at aim 1; retries divide it so the shot gets straighter
FREE_NOISE = 0.12  # radians, about 7° off the compass axis
AIM_STEP = 0.35
AIM_MAX = 5.0
PLACE_TRIES = 80
SCENE_TRIES = 40

ENTER_CHANCE = 0.25
AIM_AT_STILL = 0.7
# Beta(a, b) for P(a shape is still). A collision scene leans higher so something sits there to hit.
STILL_BETA = (3.2, 2.2)  # mean 0.59
STILL_BETA_FREE = (2.2, 2.2)  # mean 0.50
# --busy scenes: more shapes on the move, so more events happen around any one of them.
STILL_BETA_BUSY = (1.6, 3.2)  # mean 0.33
STILL_BETA_FREE_BUSY = (1.4, 3.0)  # mean 0.32

# Speeds in [STILL_SPEED, CLEAR_SPEED), or a heading this close to a diagonal, are a coin flip.
CLEAR_SPEED = 2 * STILL_SPEED
NEAR_DIAGONAL = 10.0


def unexplained(log: EventLog) -> bool:
    """The picture changed and the label has no enter, exit, or collision for it."""
    return bool(log.breaks)


def ambiguous(log: EventLog) -> bool:
    """A speed or a heading a viewer could honestly call either way."""
    return any(_dubious(vel) for vel in log.velocities())


def misaligned(log: EventLog) -> bool:
    """An enter or exit through a corner, or across an edge the shape is not heading to."""
    return bool(log.misaligned)


def _exit_directions(log: EventLog, fps: int) -> set[str]:
    return {row.get("direction", "") for moment in log.forward(fps) for row in moment["events"]
            if row["doing"] == "exiting"}


# --keep: which accepted scenes get rendered. The rest only cost a physics run.
KEEP = {
    "all": lambda log, fps: True,
    "north_and_west": lambda log, fps: {"north", "west"} <= _exit_directions(log, fps),
}


def _dubious(vel: np.ndarray) -> bool:
    speed = float(np.hypot(*vel))
    if speed < STILL_SPEED:
        return False
    if speed < CLEAR_SPEED:
        return True
    angle = np.degrees(np.atan2(vel[1], vel[0])) % 90
    return abs(angle - 45) < NEAR_DIAGONAL


def _build(kind: str, color: Color, pos: np.ndarray, vel: np.ndarray, rotation: float) -> Shape:
    extent = SHAPE / 2 if SHAPES[kind] is None else SHAPE
    scale = np.full(2, extent)
    if SHAPES[kind] is None:
        return Circle(pos, vel, scale, color)
    shape = ConvexPolygon(pos, vel, scale, kind, color)
    shape.rotation = np.float64(rotation)
    return shape


def _compass(rng: np.random.Generator, slot: int, noise: float) -> np.ndarray:
    angle = (slot % 4) * (np.pi / 2) + float(rng.normal(0, noise))
    return np.array([np.cos(angle), np.sin(angle)])


def _place(rng: np.random.Generator, span: np.ndarray, outside: bool) -> np.ndarray:
    if not outside:
        return rng.uniform(INSET, span - INSET)
    pos = rng.uniform(ALONG_EDGE[0] * span, ALONG_EDGE[1] * span)
    gap = float(rng.uniform(*OFFSCREEN))
    side = int(rng.integers(4))
    axis = side % 2
    pos[axis] = -gap if side < 2 else span[axis] + gap
    return pos


def _crowded(positions: list[np.ndarray]) -> bool:
    for i, left in enumerate(positions):
        for right in positions[i + 1 :]:
            if float(np.hypot(*(left - right))) < SEPARATION:
                return True
    return False


def sample_shapes(
    rng: np.random.Generator,
    span: np.ndarray,
    n: int,
    want_collision: bool,
    seconds: float,
    aim: float = 1.0,
    direction_slot: int = 0,
    busy: bool = False,
) -> list[Shape]:
    """Draw one scene. `aim` tightens paths toward or away from a hit."""
    # Shape and color are drawn independently per shape, with repeats, so whether a color
    # is in a clip does not depend on the other colors there. Two shapes with the same
    # shape and color would make the label ambiguous, so such a draw is redrawn.
    kinds = list(SHAPES)
    colors = list(Color)
    if busy:
        beta = STILL_BETA_BUSY if want_collision else STILL_BETA_FREE_BUSY
    else:
        beta = STILL_BETA if want_collision else STILL_BETA_FREE
    static_p = float(rng.beta(*beta))
    noise = AIM_NOISE / aim if want_collision else FREE_NOISE
    for _ in range(PLACE_TRIES):
        picked_kinds = rng.choice(len(kinds), size=n)
        picked_colors = rng.choice(len(colors), size=n)
        if len(set(zip(picked_kinds.tolist(), picked_colors.tolist()))) < n:
            continue
        draft = []
        for kind_i, color_i in zip(picked_kinds, picked_colors):
            kind, color = kinds[int(kind_i)], colors[int(color_i)]
            still = rng.random() < static_p
            outside = want_collision and not still and rng.random() < ENTER_CHANCE
            pos = _place(rng, span, outside)
            draft.append([kind, color, pos, still, outside])
        if not any(item[3] and not item[4] for item in draft):
            pinned = int(rng.integers(n))
            draft[pinned][2] = _place(rng, span, False)
            draft[pinned][3] = True
            draft[pinned][4] = False
        if not any(not item[3] for item in draft):
            resting = [i for i, item in enumerate(draft) if item[3] and not item[4]]
            choices = [i for i in range(n) if i not in resting] or resting
            draft[int(rng.choice(choices))][3] = False
        slot = direction_slot
        for index, item in enumerate(draft):
            kind, color, pos, still, outside = item
            if still:
                continue
            heading = _compass(rng, slot, noise)
            slot += 1
            if want_collision:
                others = [i for i in range(n) if i != index]
                resting = [i for i in others if draft[i][3]]
                pool = resting if resting and rng.random() < AIM_AT_STILL else others
                target = draft[int(rng.choice(pool))][2]
                dist = float(rng.uniform(*APPROACH))
                item[2] = np.asarray(target, float) - heading * dist
                speed = min(dist / (float(rng.uniform(*ARRIVE)) * seconds), MAX_SPEED)
            else:
                speed = float(rng.uniform(*CRUISE))
            item.append(heading * speed)
        positions = [item[2] for item in draft]
        if _crowded(positions):
            continue
        shapes: list[Shape] = []
        for kind, color, pos, still, outside, *rest in draft:
            vel = np.zeros(2) if still else rest[0]
            rotation = 0.0 if SHAPES[kind] is None else float(rng.uniform(0, 2 * np.pi))
            shapes.append(_build(kind, color, pos, vel, rotation))
        if any(penetration(a, b) is not None for i, a in enumerate(shapes) for b in shapes[i + 1 :]):
            continue
        return shapes
    raise RuntimeError("could not place shapes")


# Simulation() with no shapes draws a scene. Bound here so simulation.py can import at top.
simulation.sample_shapes = sample_shapes


def _clone(shapes: list[Shape]) -> list[Shape]:
    cloned = []
    for shape in shapes:
        if isinstance(shape, Circle):
            cloned.append(Circle(shape.pos.copy(), shape.vel.copy(), shape.scale.copy(), shape.color))
            continue
        cloned.append(
            ConvexPolygon(
                shape.pos.copy(), shape.vel.copy(), shape.scale.copy(), shape.type, shape.color
            )
        )
        cloned[-1].rotation = shape.rotation
    return cloned


def _produce(task: tuple[int, bool, int, str, int, int, bool, str]) -> dict:
    index, want, seed, out_dir, seconds, fps, busy, keep = task
    rng = np.random.default_rng(seed)
    span = world_span(640, 480)
    aim = 1.0
    n = _N_SHAPES_BUSY[index % 2] if busy else _N_SHAPES[index % 3]
    accepted: list[Shape] | None = None
    for _ in range(SCENE_TRIES):
        try:
            shapes = sample_shapes(rng, span, n, want, seconds, aim, direction_slot=index, busy=busy)
        except RuntimeError:
            # no placement fit this draw; count it as a failed scene, not a failed dataset
            aim = min(AIM_MAX, aim + AIM_STEP)
            continue
        probe = Simulation(shapes=_clone(shapes), width=640, height=480)
        probe.run(fps=fps, seconds=seconds)
        if (
            probe.collided() == want
            and not probe.grazes()
            and len(probe._fully_seen) == n
            and not unexplained(probe.log)
            and not ambiguous(probe.log)
            and not misaligned(probe.log)
        ):
            accepted = shapes
            break
        aim = min(AIM_MAX, aim + AIM_STEP)
    if accepted is None:
        return {"file": None, "index": index}
    if not KEEP[keep](probe.log, fps):
        return {"file": None, "index": index, "skipped": True}
    path = Path(out_dir) / f"{index:05d}.mp4"
    sim = Simulation(shapes=accepted, width=640, height=480)
    sim.run(str(path), fps=fps, seconds=seconds)
    frames = fps * seconds
    reversed_path = path.with_name(f"{index:05d}_rev.mp4")
    _write_reverse(path, reversed_path, fps)
    reversed_path.with_suffix(".json").write_text(
        json.dumps(sim.log.backward(frames, fps), indent=2)
    )
    hit = sim.collided()
    return {
        "file": path.name,
        "reverse": reversed_path.name,
        "n_shapes": len(accepted),
        "collision": hit,
    }


def _write_reverse(source: Path, dest: Path, fps: int):
    frames = [frame for frame in iio.get_reader(source)]
    with iio.get_writer(dest, fps=fps) as out:
        writer = cast(Format.Writer, out)
        for frame in reversed(frames):
            writer.append_data(frame)


def generate(
    out_dir: str,
    count: int = 50,
    seed: int = 0,
    seconds: int = 2,
    fps: int = 30,
    jobs: int = 0,
    busy: bool = False,
    keep: str = "all",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    half = count // 2
    tasks = [
        (index, index < half, seed + index + 1, str(out), seconds, fps, busy, keep) for index in range(count)
    ]
    workers = jobs or (os.cpu_count() or 1)
    with get_context("spawn").Pool(workers) as pool:
        rows = []
        skipped = 0
        for row in pool.imap_unordered(_produce, tasks):
            if row.get("skipped"):
                skipped += 1
                continue
            if row["file"] is None:
                print(f"{row['index']:05d} dropped: no clean scene", flush=True)
                continue
            label = "hit" if row["collision"] else "miss"
            print(f"{row['file']} {label} n={row['n_shapes']}", flush=True)
            rows.append(row)
    rows.sort(key=lambda row: row["file"])
    counts = {
        "true": sum(row["collision"] for row in rows),
        "false": sum(not row["collision"] for row in rows),
    }
    settings = {"seed": seed, "count": count, "busy": busy, "keep": keep, "not_kept": skipped}
    (out / "manifest.json").write_text(
        json.dumps({"counts": counts, "settings": settings, "videos": rows}, indent=2)
    )
    return {"counts": counts}
