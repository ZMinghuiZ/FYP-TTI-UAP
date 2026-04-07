#!/usr/bin/env python3
"""
Summarize ASR results from stretch x scale and post-processing sweeps.

Scans sweep/<RUN_NAME>/adv_videos_*/ directories for evaluation results
and produces a table grouped by UAP config, stretch, scale, and
post-processing variant.

Usage:
    python sweep/summarize_apply.py                   # table only
    python sweep/summarize_apply.py --csv             # also writes CSV
    python sweep/summarize_apply.py --run G3_a4-near-s8-low  # filter one UAP
    python sweep/summarize_apply.py --sort mean       # sort by mean ASR
"""

import argparse
import csv
import re
import sys
from pathlib import Path


SWEEP_ROOT = Path(__file__).resolve().parent

MODEL_PREFIXES = ["internvl", "qwen", "llava_onevision", "videollama3"]
MODEL_LABELS = ["InternVL", "Qwen3-VL", "LLaVA-OV", "VidLLaMA3"]

_RE_DIR = re.compile(
    r"^adv_videos_stretch(\d+)_scale([\d.]+)(.*)$"
)


def parse_summary_file(path):
    text = path.read_text()
    match = re.search(r"ATTACK SUCCESS RATE \(ASR\):\s*([\d.]+)%", text)
    if match:
        return float(match.group(1))
    return None


def parse_csv_results(path):
    total = 0
    succeeded = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "").strip().lower()
            if status in ("error",):
                continue
            total += 1
            if row.get("prediction", "").strip().lower() == "yes":
                succeeded += 1
    if total == 0:
        return None
    return succeeded / total * 100.0


def get_asr(result_dir, model_prefix):
    summary = result_dir / f"{model_prefix}_summary_metrics.txt"
    if summary.exists():
        asr = parse_summary_file(summary)
        if asr is not None:
            return asr
    for csv_path in sorted(result_dir.glob(f"{model_prefix}_results_*.csv")):
        asr = parse_csv_results(csv_path)
        if asr is not None:
            return asr
    return None


def discover_results(sweep_root, run_filter=None, result_type="all"):
    """Find all adv_videos_* directories with eval results.

    result_type: "all", "stretch_scale" (no postprocess), or "postprocess" only.
    """
    rows = []

    run_dirs = sorted(d for d in sweep_root.iterdir() if d.is_dir())

    for run_dir in run_dirs:
        run_name = run_dir.name
        if run_filter and run_name not in run_filter:
            continue

        for sub in sorted(run_dir.iterdir()):
            if not sub.is_dir():
                continue

            m = _RE_DIR.match(sub.name)
            if not m:
                continue

            stretch = int(m.group(1))
            scale = m.group(2)
            suffix = m.group(3)

            postprocess = "none"
            if suffix:
                postprocess = suffix.lstrip("_")

            is_pp = postprocess != "none"
            if result_type == "stretch_scale" and is_pp:
                continue
            if result_type == "postprocess" and not is_pp:
                continue

            has_results = any(
                (sub / f"{prefix}_summary_metrics.txt").exists()
                or list(sub.glob(f"{prefix}_results_*.csv"))
                for prefix in MODEL_PREFIXES
            )
            if not has_results:
                continue

            asr_vals = {}
            for prefix in MODEL_PREFIXES:
                asr_vals[prefix] = get_asr(sub, prefix)

            available = [v for v in asr_vals.values() if v is not None]
            mean_asr = sum(available) / len(available) if available else None

            rows.append({
                "run": run_name,
                "stretch": stretch,
                "scale": scale,
                "postprocess": postprocess,
                "dir": str(sub.relative_to(sweep_root)),
                "mean_asr": mean_asr,
                **{f"{p}_asr": asr_vals[p] for p in MODEL_PREFIXES},
            })

    return rows


def fmt_asr(val):
    if val is None:
        return "  --  "
    return f"{val:6.2f}%"


def print_table(rows, sort_key):
    if not rows:
        print("No results found.")
        return

    if sort_key == "mean":
        rows.sort(key=lambda r: (r["mean_asr"] or -1), reverse=True)
    elif sort_key == "run":
        rows.sort(key=lambda r: (r["run"], r["stretch"], r["scale"],
                                 r["postprocess"]))
    elif sort_key == "stretch":
        rows.sort(key=lambda r: (r["stretch"], r["scale"], r["run"],
                                 r["postprocess"]))
    else:
        rows.sort(key=lambda r: (r["run"], r["stretch"], r["scale"],
                                 r["postprocess"]))

    col_run = max(max(len(r["run"]) for r in rows), 3)
    col_pp = max(max(len(r["postprocess"]) for r in rows), 7)

    header = (
        f"{'Run':<{col_run}}  {'Str':>3}  {'Scale':>5}  "
        f"{'PostProc':<{col_pp}}  "
        f"{'InternVL':>9}  {'Qwen3-VL':>9}  "
        f"{'LLaVA-OV':>9}  {'VidLLaMA3':>9}  "
        f"{'Mean ASR':>9}"
    )
    sep = "-" * len(header)

    print()
    print("Stretch x Scale + Post-processing -- ASR Summary")
    print(sep)
    print(header)
    print(sep)

    prev_run = None
    for r in rows:
        if prev_run is not None and r["run"] != prev_run:
            print()
        prev_run = r["run"]

        print(
            f"{r['run']:<{col_run}}  {r['stretch']:>3}  {r['scale']:>5}  "
            f"{r['postprocess']:<{col_pp}}  "
            f"{fmt_asr(r.get('internvl_asr')):>9}  "
            f"{fmt_asr(r.get('qwen_asr')):>9}  "
            f"{fmt_asr(r.get('llava_onevision_asr')):>9}  "
            f"{fmt_asr(r.get('videollama3_asr')):>9}  "
            f"{fmt_asr(r['mean_asr']):>9}"
        )

    print(sep)

    complete = [r for r in rows if r["mean_asr"] is not None]
    if complete:
        best = max(complete, key=lambda r: r["mean_asr"])
        print(f"\nBest mean ASR: {best['dir']}  ({best['mean_asr']:.2f}%)")
        for prefix, label in zip(MODEL_PREFIXES, MODEL_LABELS):
            key = f"{prefix}_asr"
            with_val = [r for r in complete if r.get(key) is not None]
            if with_val:
                b = max(with_val, key=lambda r: r[key])
                print(f"  Best {label}: {b['dir']}  ({b[key]:.2f}%)")

    print(f"\nTotal: {len(rows)} result(s) across "
          f"{len(set(r['run'] for r in rows))} UAP config(s)")


def write_csv(rows, csv_path):
    fieldnames = ["run", "stretch", "scale", "postprocess", "dir"]
    for prefix in MODEL_PREFIXES:
        fieldnames.append(f"{prefix}_asr")
    fieldnames.append("mean_asr")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize stretch x scale and post-processing sweep results")
    parser.add_argument("--csv", action="store_true",
                        help="Write results to sweep/summary_apply.csv")
    parser.add_argument("--run", type=str, nargs="+", default=None,
                        help="Filter to one or more UAP run names")
    parser.add_argument("--run_list", type=str, default=None,
                        help="File with one UAP run name per line "
                             "(e.g. sweep/eval_top_list.txt)")
    parser.add_argument("--type", choices=["all", "stretch_scale", "postprocess"],
                        default="all", dest="result_type",
                        help="Filter result type: all, stretch_scale only, "
                             "or postprocess only (default: all)")
    parser.add_argument("--sort", choices=["run", "mean", "stretch"],
                        default="run",
                        help="Sort order (default: run)")
    args = parser.parse_args()

    run_filter = args.run
    if args.run_list:
        list_path = Path(args.run_list)
        if not list_path.exists():
            print(f"ERROR: {args.run_list} not found.")
            sys.exit(1)
        file_names = [
            line.strip() for line in list_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if run_filter:
            run_filter = list(set(run_filter + file_names))
        else:
            run_filter = file_names

    rows = discover_results(SWEEP_ROOT, run_filter=run_filter,
                            result_type=args.result_type)

    if not rows:
        print(f"No adv_videos_stretch*_scale* directories with results found "
              f"under {SWEEP_ROOT}/")
        if args.run:
            print(f"  (filtered to run: {args.run})")
        sys.exit(1)

    print_table(rows, sort_key=args.sort)

    if args.csv:
        write_csv(rows, SWEEP_ROOT / "summary_apply.csv")


if __name__ == "__main__":
    main()
