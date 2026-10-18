import argparse
import random

from .simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Write one random shape video")
    parser.add_argument("-o", "--output", default="shapes.mp4")
    args = parser.parse_args()
    Simulation(seed=random.randrange(2**31)).run(args.output)


if __name__ == "__main__":
    main()
