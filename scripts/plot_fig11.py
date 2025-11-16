#!/usr/bin/env python3
import argparse
import csv
import os
from matplotlib import pyplot as plt

def read_series(path):
    xs, ys = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["val_size"]))
            ys.append(float(row["throughput_ops_per_s"]))
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]

def add_panel(ax, base_path, val_path, title):
    if base_path and os.path.isfile(base_path):
        x_b, y_b = read_series(base_path)
        ax.plot(x_b, y_b, marker="D", label="H-Rocks", color="#1f77b4")
    if val_path and os.path.isfile(val_path):
        x_v, y_v = read_series(val_path)
        ax.plot(x_v, y_v, marker="o", label="H-Rocks with values", color="#d62728")
    ax.set_xlabel("Value sizes")
    ax.set_ylabel("Throughput (ops/s)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puts-base", required=True, help="CSV for baseline puts")
    ap.add_argument("--puts-with-values", required=True, help="CSV for puts with values")
    ap.add_argument("--gets-base", help="CSV for baseline gets")
    ap.add_argument("--gets-with-values", help="CSV for gets with values")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    add_panel(axes[0], args.puts_base, args.puts_with_values, "(a) PUTs (R50:W50)")

    if args.gets_base and args.gets_with_values and os.path.isfile(args.gets_base) and os.path.isfile(args.gets_with_values):
        add_panel(axes[1], args.gets_base, args.gets_with_values, "(b) GETs (R90:W10)")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "GET data missing", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Wrote {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
