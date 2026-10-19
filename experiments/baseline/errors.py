"""Where the observer goes wrong: direction confusions and what missed events turn into."""
import json
import sys
from collections import Counter
from pathlib import Path

from genai_vlm_jailbreak.generator.events import HEADINGS
from genai_vlm_jailbreak.score import Row, Score


def gap(a, b):
    if a not in HEADINGS or b not in HEADINGS:
        return "invalid"
    d = abs(HEADINGS.index(a) - HEADINGS.index(b)) % 4
    return {0: "0", 1: "90", 2: "180"}[min(d, 4 - d)]


def main():
    run, data = Path(sys.argv[1]), Path(sys.argv[2])
    start_gap, event_gap, missed_as, axis = Counter(), Counter(), Counter(), Counter()
    for line in run.open():
        d = json.loads(line)
        truth = json.loads((data / f"{d['clip']}.json").read_text())
        pred = Score.parse(d["reply"])
        if not pred:
            continue
        guessed = {row.who: row for row in Row.start(pred)}
        for row in Row.start(truth):
            guess = guessed.get(row.who)
            if row.doing == "moving" and guess and guess.doing == "moving":
                g = gap(row.direction, guess.direction)
                start_gap[g] += 1
                if g == "180":
                    axis[(row.direction, guess.direction)] += 1
        seen = Row.events(pred)
        for true in Row.events(truth):
            same = [row for row in seen if row.who == true.who and row.doing == true.doing]
            if same:
                if true.direction is not None:
                    event_gap[(true.doing, gap(true.direction, same[0].direction))] += 1
                continue
            other = [row.doing for row in seen if row.who == true.who]
            missed_as[(true.doing, tuple(sorted(set(other))) or ("in t=0" if true.who in guessed else "absent"))] += 1
    print("start direction error (degrees):", dict(start_gap))
    print("180-degree flips, true->pred:", axis.most_common(8))
    print("event direction error:", sorted(event_gap.items()))
    print("missed events, what the model said instead:", missed_as.most_common(15))


if __name__ == "__main__":
    main()
