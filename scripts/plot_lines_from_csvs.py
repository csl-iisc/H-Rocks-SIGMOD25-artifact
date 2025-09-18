#!/usr/bin/env python3
import argparse, csv, os
from matplotlib import pyplot as plt

def read_xy(csv_path, pick_val_size=None):
    xs, ys = [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        headers = [h.strip() for h in r.fieldnames]
        has_val = "val_size" in headers
        for row in r:
            if has_val and pick_val_size is not None and str(row.get("val_size","")).strip() != str(pick_val_size):
                continue
            x = float(row["size"])
            y = float(row.get("throughput_ops_per_s") or row.get("throughput"))
            xs.append(x); ys.append(y)
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--xlabel", default="Size")
    ap.add_argument("--ylabel", default="Throughput (ops/s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlog", action="store_true")
    ap.add_argument("--val-size", type=int, default=None,
                    help="If a CSV has val_size column (e.g., Viper GETS), pick this one.")
    ap.add_argument("--series", action="append", required=True,
                    help="Format: /path/to.csv:Label")
    # NEW:
    ap.add_argument("--cap-to-x", action="store_true",
                    help="Clamp y so y<=x (useful for sustained throughput vs arrival-rate).")
    ap.add_argument("--y-mops", action="store_true",
                    help="Plot Y in Mops/s (divide by 1e6) and adjust ylabel automatically.")
    args = ap.parse_args()

    plt.figure()
    for item in args.series:
        path, label = item.split(":", 1)
        xs, ys = read_xy(path, pick_val_size=args.val_size)
        if not xs:
            continue

        # cap throughput to offered load (per-point)
        if args.cap_to_x:
            ys = [min(y, x) for x, y in zip(xs, ys)]

        # convert to Mops/s
        if args.y_mops:
            ys = [y/1e6 for y in ys]

        plt.plot(xs, ys, marker="o", label=label)

    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel("Sustained T.put (Mops/sec)" if args.y_mops else args.ylabel)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    if args.xlog:
        plt.xscale("log")

    plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
