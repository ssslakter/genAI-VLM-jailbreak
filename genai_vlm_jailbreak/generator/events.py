import numpy as np

from .shapes import Circle, Shape

# four compass directions, +x is east and +y is south, matching the video
_HEADINGS = ("east", "south", "west", "north")

# 40 px/s on a 480 px tall frame: slower than this reads as standing still
STILL_SPEED = 40 / (480 / (1 / 0.12))
# speeds in [STILL_SPEED, 2 * STILL_SPEED) and headings this close to a diagonal are
# ambiguous to a viewer, so scenes containing them are resampled
AMBIGUOUS_DEGREES = 10.0


def _kind(shape: Shape) -> str:
    return "circle" if isinstance(shape, Circle) else shape.type


def _heading(vel: np.ndarray) -> str:
    return _HEADINGS[int(np.round(np.atan2(vel[1], vel[0]) / (np.pi / 2))) % 4]


def ambiguous(vel: np.ndarray) -> bool:
    speed = float(np.hypot(*vel))
    if speed < STILL_SPEED:
        return False
    if speed < 2 * STILL_SPEED:
        return True
    angle = np.degrees(np.atan2(vel[1], vel[0])) % 90
    return abs(angle - 45) < AMBIGUOUS_DEGREES


def _figure(shape: Shape) -> dict[str, str]:
    return {"shape": _kind(shape), "color": shape.color.name.lower()}


_OPPOSITE_DOING = {"entering": "exiting", "exiting": "entering"}


def _flip_direction(direction: str) -> str:
    return _HEADINGS[(_HEADINGS.index(direction) + 2) % 4]


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
    direction = _motion(shape.vel)
    if shown and not was_shown:
        return "entering", direction
    if was_shown and not shown:
        return "exiting", direction
    if direction is None:
        return "still", None
    return "moving", direction


_EVENTS = ("entering", "exiting", "colliding")


def _motion(vel: np.ndarray) -> str | None:
    return None if float(np.hypot(*vel)) < STILL_SPEED else _heading(vel)


class SparseLog:
    """The starting state, then only entering, exiting and colliding, with frame numbers.

    Each record keeps the velocity before and after its frame, so the same record can be
    told forward or backward: played backward, a shape leaves a collision with the
    reverse of the velocity it came in with.
    """

    def __init__(self):
        self.start: list[dict] = []
        self.end: list[dict] = []
        self.records: list[dict] = []
        self.unexplained: list[tuple[int, str]] = []
        self.ambiguous: list[tuple[int, str]] = []
        self._motion: dict[int, str | None] = {}

    @staticmethod
    def _state(shape: Shape, vel: np.ndarray) -> dict:
        row = {**_figure(shape)}
        direction = _motion(vel)
        row["doing"] = "still" if direction is None else "moving"
        if direction is not None:
            row["direction"] = direction
        return row

    def observe(
        self,
        frame: int,
        shapes: list[Shape],
        before: dict[int, np.ndarray],
        was_shown: dict[int, bool],
        shown: dict[int, bool],
        collisions: list[tuple[Shape, Shape]],
        hidden_hits: bool,
    ):
        partner = {}
        for left, right in collisions:
            partner[id(left)] = right
            partner[id(right)] = left
        if frame == 0:
            self.start = [self._state(s, s.vel) for s in shapes if id(s) in shown]
            self.ambiguous += [(0, _kind(s)) for s in shapes if id(s) in shown and ambiguous(s.vel)]
        if hidden_hits:
            self.unexplained.append((frame, "collision with a shape that is not shown"))
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
                self.unexplained.append((frame, f"{_kind(shape)} changed motion"))
            self._motion[mark] = now
            if doing is None or frame == 0:
                continue
            if ambiguous(before[mark]) or ambiguous(shape.vel):
                self.ambiguous.append((frame, f"{_kind(shape)} {doing}"))
            record = {"frame": frame, "figure": _figure(shape), "doing": doing,
                      "before": before[mark].tolist(), "after": shape.vel.tolist()}
            if doing == "colliding":
                record["with"] = _figure(partner[mark])
            self.records.append(record)

    def finish(self, shapes: list[Shape], shown: dict[int, bool]):
        self.end = [self._state(s, s.vel) for s in shapes if id(s) in shown]
        self.ambiguous += [(-1, _kind(s)) for s in shapes if id(s) in shown and ambiguous(s.vel)]

    def forward(self, fps: int) -> list[dict]:
        return _tell(self.start, self.records, lambda r: r["frame"], fps, backward=False)

    def backward(self, frames: int, fps: int) -> list[dict]:
        start = [_flip_row(row) for row in self.end]
        # a crossing between frames f-1 and f happens between last-f and last-f+1 played back,
        # a contact on frame f is on frame last-f
        def frame_of(record: dict) -> int:
            return frames - record["frame"] - (record["doing"] == "colliding")

        return _tell(start, self.records, frame_of, fps, backward=True)


def _row(record: dict, backward: bool) -> dict:
    doing = record["doing"]
    vel = np.asarray(record["after"])
    if backward:
        doing = _OPPOSITE_DOING.get(doing, doing)
        vel = -np.asarray(record["before"])
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
