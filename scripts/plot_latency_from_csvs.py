#!/usr/bin/env python3
import argparse, csv, os
from matplotlib import pyplot as plt

def read_size_throughput(csv_path, pick_val_size=None):
    xs, thr = [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        headers = [h.strip() for h in r.fieldnames]
        has_val = "val_size" in headers
        for row in r:
            if has_val and pick_val_size is not None and str(row.get("val_size","")).strip() != str(pick_val_size):
                continue
            x = float(row["size"])
            y = float(row.get("throughput_ops_per_s") or row.get("throughput"))
            xs.append(x); thr.append(y)
    pairs = sorted(zip(xs, thr), key=lambda p: p[0])
    xs_sorted = [p[0] for p in pairs]
    thr_sorted = [p[1] for p in pairs]
    return xs_sorted, thr_sorted

def to_latency_ms(ops_per_s):
    # latency (ms/op) = 1000 / throughput(ops/s)
    return [(1000.0/x if x > 0 else 0.0) for x in ops_per_s]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--xlabel", default="Arrival request rate (ops/sec)")
    ap.add_argument("--ylabel", default="Latency (msecs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlog", action="store_true")
    ap.add_argument("--val-size", type=int, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="ymin ymax")
    ap.add_argument("--series", action="append", required=True, help="CSV:Label")
    args = ap.parse_args()

    plt.figure()
    for item in args.series:
        path, label = item.split(":", 1)
        xs, thr = read_size_throughput(path, pick_val_size=args.val_size)
        if not xs:
            continue
        lat_ms = to_latency_ms(thr)
        plt.plot(xs, lat_ms, marker="o", label=label)

    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    if args.xlog:
        plt.xscale("log")
    if args.ylim:
        plt.ylim(args.ylim)
    plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
