print("SCRIPT STARTED")

import splitfolders
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True)
parser.add_argument("--dest", type=str, default="data")
parser.add_argument("--ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1])
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

print(f"Splitting {args.src} -> {args.dest} with ratio {args.ratio}")

splitfolders.ratio(args.src, output=args.dest, seed=args.seed, ratio=tuple(args.ratio))

print("DONE SPLITTING.")
