# Triggers

How many generated clips realize each rung of a trigger ladder, and how many are near-misses?

This does not call a model. `count.py` reads the ground-truth logs and prints, per dataset directory, the share of clips in each bin:

- presence of a circle, the color red, a collision
- conjunction: a red circle, versus red and circle split across shapes
- set: a collision and an exit, versus only one of them
- sequence: a collision and then an exit, versus the reverse order or the two within 0.1 s

```bash
pixi run python experiments/triggers/count.py dataset
```

## Results

None checked in yet.
