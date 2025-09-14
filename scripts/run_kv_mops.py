#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
import matplotlib.pyplot as plt

KV_ORDER = [(8,8),(16,32),(16,128),(32,256),(64,128),(128,1024)]
KV_LABELS = [f"{k}/{v}" for k,v in KV_ORDER]

def load_series(csv_path: Path):
    """
    Accept either:
      - size,k,v,throughput_ops_per_s
      - kv,count,throughput_mops
    Returns dict: { "8/8": mops, ... }
    If multiple counts present, chooses the largest 'size' (count).
    """
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    # Detect schema
    has_ops = all(h in reader.fieldnames for h in ("size","k","v","throughput_ops_per_s"))
    has_mops = all(h in reader.fieldnames for h in ("kv","count","throughput_mops"))

    data = {}
    if has_ops:
        # Group by kv, pick largest size
        grouped = {}
        for r in rows:
            k = int(r["k"]); v = int(r["v"])
            size = int(r["size"])
            ops = float(r["throughput_ops_per_s"]) if r["throughput_ops_per_s"] else 0.0
            kv = f"{k}/{v}"
            if kv not in grouped or size > grouped[kv][0]:
                grouped[kv] = (size, ops/1e6)  # -> Mops/s
        for kv, (_, mops) in grouped.items():
            data[kv] = mops
    elif has_mops:
        grouped = {}
        for r in rows:
            kv = r["kv"]
            cnt = int(r["count"])
            mops = float(r["throughput_mops"]) if r["throughput_mops"] else 0.0
            if kv not in grouped or cnt > grouped[kv][0]:
                grouped[kv] = (cnt, mops)
        for kv, (_, mops) in grouped.items():
            data[kv] = mops
    else:
        raise ValueError(f"Unrecognized CSV schema: {csv_path}")
    return data

def plot_panel(series_list, title: str, out_png: Path):
    xs = list(range(len(KV_LABELS)))
    plt.figure()
    for label, data in series_list:
        ys = [data.get(kv, 0.0) for kv in KV_LABELS]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xticks(xs, KV_LABELS)
    plt.xlabel("KV pair sizes")
    plt.ylabel("Throughput (Mops/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Wrote {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puts", action="append", default=[], help="CSV:Label for PUTs (repeatable)")
    ap.add_argument("--gets", action="append", default=[], help="CSV:Label for GETs (repeatable)")
    ap.add_argument("--title_puts", default="PUTs with varying key-value sizes")
    ap.add_argument("--title_gets", default="GETs with varying key-value sizes")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    puts_series = []
    for s in args.puts:
        path,label = s.split(":",1)
        data = load_series(Path(path))
        puts_series.append((label, data))

    gets_series = []
    for s in args.gets:
        path,label = s.split(":",1)
        data = load_series(Path(path))
        gets_series.append((label, data))

    if puts_series:
        plot_panel(puts_series, args.title_puts, out_dir / "fig10_puts.png")
    if gets_series:
        plot_panel(gets_series, args.title_gets, out_dir / "fig10_gets.png")

if __name__ == "__main__":
    main()
