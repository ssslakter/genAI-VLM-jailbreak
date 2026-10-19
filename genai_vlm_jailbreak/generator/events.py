import numpy as np

from .shapes import Circle, Shape

_HEADINGS = (
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
    "north",
    "northeast",
)


def _kind(shape: Shape) -> str:
    return "circle" if isinstance(shape, Circle) else shape.type


def _heading(vel: np.ndarray) -> str:
    # +x is east and +y is south, matching the video
    return _HEADINGS[int(np.round(np.atan2(vel[1], vel[0]) / (np.pi / 4))) % 8]


def _figure(shape: Shape) -> dict[str, str]:
    return {"shape": _kind(shape), "color": shape.color.name.lower()}


_OPPOSITE_DOING = {"entering": "exiting", "exiting": "entering"}


def _flip_direction(direction: str) -> str:
    return _HEADINGS[(_HEADINGS.index(direction) + 4) % 8]


def _flip_row(row: dict) -> dict:
    flipped = dict(row)
    flipped["doing"] = _OPPOSITE_DOING.get(row["doing"], row["doing"])
    if "direction" in row:
        flipped["direction"] = _flip_direction(row["direction"])
    if "with" in row:
        flipped["with"] = dict(row["with"])
    return flipped


def reverse_log(events: list[dict], frames: int, fps: int) -> list[dict]:
    """Play the change-log backwards. A state lasts until the next timestamp."""
    last = frames - 1
    moments = []
    for index, moment in enumerate(events):
        if index + 1 == len(events):
            start = 0
        else:
            start = last - int(round(events[index + 1]["time"] * fps)) + 1
        moments.append(
            {"time": round(start / fps, 3), "events": [_flip_row(row) for row in moment["events"]]}
        )
    moments.sort(key=lambda moment: moment["time"])
    return moments


class EventLog:
    """One record per moment something changes. Each record lists every shape."""

    def __init__(self):
        self.events: list[dict] = []
        self._signature: tuple | None = None

    def observe(
        self,
        time: float,
        shapes: list[Shape],
        was_shown: dict[int, bool],
        shown: dict[int, bool],
        collisions: list[tuple[Shape, Shape]],
    ):
        partner = {}
        for left, right in collisions:
            partner[id(left)] = right
            partner[id(right)] = left
        rows = []
        signature = []
        for shape in shapes:
            mark = id(shape)
            if mark not in shown:
                continue
            doing, direction = _activity(shape, was_shown.get(mark, False), shown[mark])
            row = {**_figure(shape), "doing": doing}
            if direction is not None:
                row["direction"] = direction
            other = partner.get(mark)
            if other is not None and id(other) in shown:
                row["doing"] = "colliding"
                row["with"] = _figure(other)
            rows.append(row)
            signature.append((mark, doing, direction, shown[mark]))
        signature = tuple(signature)
        if not rows:
            if self._signature in (None, ()):
                return
            signature = ()
        if signature == self._signature and not collisions:
            return
        self._signature = signature
        self.events.append({"time": round(time, 3), "events": rows})


def _activity(shape: Shape, was_shown: bool, shown: bool) -> tuple[str, str | None]:
    direction = None if float(np.hypot(*shape.vel)) < 1e-2 else _heading(shape.vel)
    if shown and not was_shown:
        return "entering", direction
    if was_shown and not shown:
        return "exiting", direction
    if direction is None:
        return "still", None
    return "moving", direction
