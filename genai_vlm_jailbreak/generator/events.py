import numpy as np

from .shapes import SHAPES, Circle, Shape

# Compass words in the label. +x is east and +y is south, matching the video.
HEADINGS = ("east", "south", "west", "north")

# World speed is frame-heights per second. Slower than this is "still".
STILL_SPEED = 1 / 12


def _kind(shape: Shape) -> str:
    if isinstance(shape, Circle):
        return next(name for name, mesh in SHAPES.items() if mesh is None)
    return shape.type


def _heading(vel: np.ndarray) -> str:
    return HEADINGS[int(np.round(np.atan2(vel[1], vel[0]) / (np.pi / 2))) % 4]


def _motion(vel: np.ndarray) -> str | None:
    return None if float(np.hypot(*vel)) < STILL_SPEED else _heading(vel)


def _figure(shape: Shape) -> dict[str, str]:
    return {"shape": _kind(shape), "color": shape.color.name.lower()}


def _state(shape: Shape, vel: np.ndarray) -> dict:
    row = {**_figure(shape)}
    direction = _motion(vel)
    row["doing"] = "still" if direction is None else "moving"
    if direction is not None:
        row["direction"] = direction
    return row


_OPPOSITE = {"entering": "exiting", "exiting": "entering"}


def _flip_direction(direction: str) -> str:
    return HEADINGS[(HEADINGS.index(direction) + 2) % 4]


def _flip_row(row: dict) -> dict:
    flipped = dict(row)
    flipped["doing"] = _OPPOSITE.get(row["doing"], row["doing"])
    if "direction" in row:
        flipped["direction"] = _flip_direction(row["direction"])
    if "with" in row:
        flipped["with"] = dict(row["with"])
    return flipped


class EventLog:
    """Who is visible at the start, then each enter, exit, and collision.

    A record keeps the velocity before and after its frame, so the same record
    reads backward: leaving a hit, the shape takes the reverse of the velocity
    it arrived with. `breaks` are motion changes this label has no line for.
    """

    def __init__(self):
        self.start: list[dict] = []
        self.end: list[dict] = []
        self.records: list[dict] = []
        self.breaks: list[str] = []
        self._motion: dict[int, str | None] = {}
        self._start_vel: list[np.ndarray] = []
        self._end_vel: list[np.ndarray] = []

    def observe(
        self,
        frame: int,
        shapes: list[Shape],
        before: dict[int, np.ndarray],
        was_shown: dict[int, bool],
        shown: dict[int, bool],
        collisions: list[tuple[Shape, Shape]],
        hidden_hit: bool,
    ):
        partner = {}
        for left, right in collisions:
            partner[id(left)] = right
            partner[id(right)] = left
        if frame == 0:
            visible = [s for s in shapes if id(s) in shown]
            self.start = [_state(s, s.vel) for s in visible]
            self._start_vel = [s.vel.copy() for s in visible]
        if hidden_hit:
            self.breaks.append("collision with a shape that is not shown")
        for shape in shapes:
            mark = id(shape)
            if mark not in shown:
                self._motion.pop(mark, None)
                continue
            now = _motion(shape.vel)
            if was_shown.get(mark, False) and not shown[mark]:
                doing = "exiting"
            elif shown[mark] and not was_shown.get(mark, False):
                doing = "entering"
            else:
                doing = None
            if mark in partner:
                doing = "colliding"
            if doing is None and mark in self._motion and self._motion[mark] != now:
                self.breaks.append(f"{_kind(shape)} changed motion")
            self._motion[mark] = now
            if doing is None or frame == 0:
                continue
            record = {
                "frame": frame,
                "figure": _figure(shape),
                "doing": doing,
                "before": before[mark].copy(),
                "after": shape.vel.copy(),
            }
            if doing == "colliding":
                record["with"] = _figure(partner[mark])
            self.records.append(record)

    def finish(self, shapes: list[Shape], shown: dict[int, bool]):
        visible = [s for s in shapes if id(s) in shown]
        self.end = [_state(s, s.vel) for s in visible]
        self._end_vel = [s.vel.copy() for s in visible]

    def velocities(self) -> list[np.ndarray]:
        found = [*self._start_vel, *self._end_vel]
        for record in self.records:
            found.append(np.asarray(record["before"], float))
            found.append(np.asarray(record["after"], float))
        return found

    def forward(self, fps: int) -> list[dict]:
        return _tell(self.start, self.records, lambda record: record["frame"], fps, backward=False)

    def backward(self, frames: int, fps: int) -> list[dict]:
        # A crossing between frames f-1 and f, played backward, shows up on the
        # frame that reveals the new state. A contact on frame f stays on that picture.
        def frame_of(record: dict) -> int:
            return frames - record["frame"] - (record["doing"] == "colliding")

        return _tell([_flip_row(row) for row in self.end], self.records, frame_of, fps, backward=True)


def _row(record: dict, backward: bool) -> dict:
    doing = record["doing"]
    vel = np.asarray(record["after"], float)
    if backward:
        doing = _OPPOSITE.get(doing, doing)
        vel = -np.asarray(record["before"], float)
    row = {**record["figure"], "doing": doing}
    direction = _motion(vel)
    if direction is not None:
        row["direction"] = direction
    if "with" in record:
        row["with"] = dict(record["with"])
    return row


def _tell(start: list[dict], records: list[dict], frame_of, fps: int, backward: bool) -> list[dict]:
    moments: dict[int, list[dict]] = {}
    for record in records:
        moments.setdefault(frame_of(record), []).append(_row(record, backward))
    out = [{"time": 0.0, "events": start}]
    for frame in sorted(moments):
        out.append({"time": round(frame / fps, 3), "events": moments[frame]})
    return out
