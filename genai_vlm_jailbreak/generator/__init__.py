import argparse

from .dataset import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a balanced set of shape videos")
    parser.add_argument("-o", "--out", default="dataset")
    parser.add_argument("-n", "--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=0, help="worker processes, 0 uses every core")
    parser.add_argument("--busy", action="store_true", help="3-4 shapes, fewer of them still")
    parser.add_argument("--keep", choices=["all", "north_and_west"], default="all",
                        help="render only scenes that pass this filter")
    args = parser.parse_args()
    print(generate(args.out, args.count, args.seed, args.seconds, args.fps, args.jobs, args.busy, args.keep))


if __name__ == "__main__":
    main()
