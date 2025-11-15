#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
import matplotlib.pyplot as plt

KV_ORDER = [(8,8),(16,32),(16,128),(32,256),(64,128),(128,1024)]
KV_LABELS = [f"{k}/{v}" for k,v in KV_ORDER]

def load_series(csv_path: Path):
    """
    Accept either:
      - size,k,v,throughput_ops_per_s (headered)
      - kv,count,throughput_mops (headered or headerless)
    Returns dict: { "8/8": mops, ... }
    If multiple counts present, chooses the largest 'size' (count).
    """
    with csv_path.open() as f:
        rows = [ [col.strip() for col in row] for row in csv.reader(f) if row ]
    if not rows:
        return {}

    header = [col.lower() for col in rows[0]]
    header_map = {name: idx for idx, name in enumerate(header)}
    data_rows = rows[1:]

    def has_fields(fields):
        return all(field in header_map for field in fields)

    data = {}
    if has_fields(("size","k","v","throughput_ops_per_s")):
        grouped = {}
        for r in data_rows:
            k = int(r[header_map["k"]]); v = int(r[header_map["v"]])
            size = int(r[header_map["size"]])
            ops_str = r[header_map["throughput_ops_per_s"]]
            ops = float(ops_str) if ops_str else 0.0
            kv = f"{k}/{v}"
            if kv not in grouped or size > grouped[kv][0]:
                grouped[kv] = (size, ops / 1e6)  # convert to Mops/s
        for kv, (_, mops) in grouped.items():
            data[kv] = mops
    elif has_fields(("kv","count","throughput_mops")):
        grouped = {}
        for r in data_rows:
            kv = r[header_map["kv"]]
            cnt = int(r[header_map["count"]])
            mops_str = r[header_map["throughput_mops"]]
            mops = float(mops_str) if mops_str else 0.0
            if kv not in grouped or cnt > grouped[kv][0]:
                grouped[kv] = (cnt, mops)
        for kv, (_, mops) in grouped.items():
            data[kv] = mops
    else:
        # Headerless CSV assumed to be kv,count,throughput_mops
        grouped = {}
        for r in rows:
            if len(r) < 3:
                raise ValueError(f"Unrecognized CSV schema (too few columns): {csv_path}")
            kv, cnt, mops = r[0], int(r[1]), float(r[2]) if r[2] else 0.0
            if kv not in grouped or cnt > grouped[kv][0]:
                grouped[kv] = (cnt, mops)
        for kv, (_, mops) in grouped.items():
            data[kv] = mops
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
