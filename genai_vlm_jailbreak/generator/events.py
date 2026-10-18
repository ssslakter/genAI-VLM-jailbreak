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


class EventLog:
    def __init__(self):
        self.events: list[dict] = []
        self._heading: dict[int, str] = {}
        self._contacts: set[tuple[int, int]] = set()

    def motions(self, time: float, shapes: list[Shape]):
        for shape in shapes:
            if float(np.hypot(*shape.vel)) < 1e-3:
                continue
            heading = _heading(shape.vel)
            if self._heading.get(id(shape)) == heading:
                continue
            self._heading[id(shape)] = heading
            self.events.append(
                {"time": round(time, 3), "event": "motion", **_figure(shape), "direction": heading}
            )

    def collision(self, time: float, a: Shape, b: Shape):
        key = tuple(sorted((id(a), id(b))))
        if key in self._contacts:
            return
        self._contacts.add(key)
        self.events.append(
            {
                "time": round(time, 3),
                "event": "collision",
                "shapes": [_figure(a), _figure(b)],
            }
        )

    def finish_contacts(self, touching: set[tuple[int, int]]):
        self._contacts &= touching
