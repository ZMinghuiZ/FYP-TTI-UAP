#!/usr/bin/env python3
"""
Summarize temporal ablation results vs baselines (G4, S13, etc.).

Reads in-order baseline ASR from summary_apply_w_no_temporal.csv and
ablation results from EXP_* directories, then prints per-source
comparison tables with deltas relative to each source's own baseline.

Handles both legacy naming (EXP_shuffle_s42) and tagged naming
(EXP_S13_shuffle_s42_s12) from the unified ablation script.

Usage:
    python sweep/summarize_temporal_ablation.py
    python sweep/summarize_temporal_ablation.py --csv
    python sweep/summarize_temporal_ablation.py --source S13
    python sweep/summarize_temporal_ablation.py --source S13 --stretch 12 --csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


SWEEP_ROOT = Path(__file__).resolve().parent

MODEL_PREFIXES = ["internvl", "qwen", "llava_onevision", "videollama3"]
MODEL_LABELS = ["InternVL", "Qwen3-VL", "LLaVA-OV", "VidLLaMA3"]

KNOWN_VARIANTS = [
    "shuffle_s42", "shuffle_s123", "shuffle_s999",
    "static_frame0", "static", "reversed",
]

RUN_TO_SOURCE = {
    "G4_a4-near-s8-high": "G4",
    "S13_no-temporal-small": "S13",
}


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


def get_asr(search_dir, model_prefix, stretch_filter=None):
    """Find ASR for a model in a directory (checks dir + adv_videos*/)."""
    candidates = []
    if search_dir.is_dir():
        subdirs = sorted(
            (d for d in search_dir.iterdir()
             if d.is_dir() and d.name.startswith("adv_videos")),
            key=lambda p: p.stat().st_mtime, reverse=True)
        if stretch_filter:
            tag = f"stretch{stretch_filter}"
            subdirs = [d for d in subdirs if tag in d.name]
        candidates.extend(subdirs)
    candidates.append(search_dir)

    for d in candidates:
        if not d.is_dir():
            continue
        summary = d / f"{model_prefix}_summary_metrics.txt"
        if summary.exists():
            asr = parse_summary_file(summary)
            if asr is not None:
                return asr
        for csv_path in sorted(d.glob(f"{model_prefix}_results_*.csv")):
            asr = parse_csv_results(csv_path)
            if asr is not None:
                return asr
    return None


def _float_or_none(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _mean_asr(asrs):
    available = [v for v in asrs.values() if v is not None]
    return sum(available) / len(available) if available else None


def parse_exp_dirname(name):
    """Parse EXP_* directory name into (source, variant, stretch|None).

    Returns None for unrecognised names (e.g. EXP_event_verify).

    Examples:
        EXP_shuffle_s42           -> ("G4", "shuffle_s42", None)
        EXP_S13_shuffle_s42_s12   -> ("S13", "shuffle_s42", "12")
        EXP_reversed_s24          -> ("G4", "reversed", "24")
    """
    if not name.startswith("EXP_"):
        return None

    rest = name[4:]

    stretch = None
    m = re.match(r"^(.+)_s(\d+)$", rest)
    if m:
        candidate, stretch_candidate = m.group(1), m.group(2)
        for v in sorted(KNOWN_VARIANTS, key=len, reverse=True):
            if candidate == v or candidate.endswith("_" + v):
                rest = candidate
                stretch = stretch_candidate
                break

    for v in sorted(KNOWN_VARIANTS, key=len, reverse=True):
        if rest == v:
            return ("G4", v, stretch)
        if rest.endswith("_" + v):
            tag = rest[:-(len(v) + 1)]
            return (tag, v, stretch)

    return None


def load_baselines():
    """Load in-order baselines from summary_apply_w_no_temporal.csv.

    Returns dict keyed by (source, stretch_str) -> {model_asr: float}.
    """
    baselines = {}
    csv_path = SWEEP_ROOT / "summary_apply_w_no_temporal.csv"
    if not csv_path.exists():
        csv_path = SWEEP_ROOT / "summary_G4_stretch_scale.csv"
    if not csv_path.exists():
        return baselines

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stretch = row.get("stretch", "")
            scale = row.get("scale", "")
            run = row.get("run", "")
            if scale != "1":
                continue
            source = None
            for prefix, tag in RUN_TO_SOURCE.items():
                if run.startswith(prefix):
                    source = tag
                    break
            if source is None:
                continue
            asrs = {
                "internvl_asr": _float_or_none(row.get("internvl_asr")),
                "qwen_asr": _float_or_none(row.get("qwen_asr")),
                "llava_onevision_asr": _float_or_none(
                    row.get("llava_onevision_asr")),
                "videollama3_asr": _float_or_none(
                    row.get("videollama3_asr")),
            }
            baselines[(source, stretch)] = asrs
    return baselines


def print_table(source, stretch, baseline_asrs, ablation_rows):
    """Print one comparison table for a (source, stretch) group."""
    label = f"{source} (stretch={stretch})" if stretch else source

    rows_to_print = []

    if baseline_asrs:
        row = {"name": f"{source} in-order (baseline)"}
        row.update(baseline_asrs)
        row["mean_asr"] = _mean_asr(baseline_asrs)
        rows_to_print.append(row)

    rows_to_print.extend(ablation_rows)

    if not rows_to_print:
        return

    col_w = max(max(len(r["name"]) for r in rows_to_print), 10)

    def fmt(val):
        return f"{val:6.2f}%" if val is not None else "   -- "

    def delta(val, base):
        if val is None or base is None:
            return ""
        return f"({val - base:+.1f})"

    print()
    print(f"Temporal Order Ablation — {label}")
    print("=" * (col_w + 75))
    header = (f"{'Condition':<{col_w}}  "
              f"{'InternVL':>9}  {'Qwen3-VL':>9}  "
              f"{'LLaVA-OV':>9}  {'VidLLaMA3':>9}  {'Mean':>9}")
    print(header)
    print("-" * len(header))

    for r in rows_to_print:
        is_baseline = baseline_asrs and r is rows_to_print[0]
        parts = [f"{r['name']:<{col_w}}"]
        for prefix in MODEL_PREFIXES:
            key = f"{prefix}_asr"
            val = r.get(key)
            base = baseline_asrs.get(key) if baseline_asrs else None
            d = "" if is_baseline else delta(val, base)
            parts.append(f"{fmt(val):>9}{d:>7}")
        parts.append(f"{fmt(r.get('mean_asr')):>9}")
        print("  ".join(parts))

    print("-" * len(header))
    if baseline_asrs:
        print(f"  Deltas relative to {source} in-order baseline.")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize temporal ablation vs baselines")
    parser.add_argument("--csv", action="store_true",
                        help="Write results to CSV")
    parser.add_argument("--stretch", type=str, default=None,
                        help="Only show results for this stretch value "
                             "(e.g. --stretch 12). Legacy dirs without "
                             "stretch in name are assumed stretch=12.")
    parser.add_argument("--source", type=str, default=None,
                        help="Only show results for this source tag "
                             "(e.g. --source S13)")
    args = parser.parse_args()

    baselines = load_baselines()
    if not baselines:
        print("WARNING: No baseline CSVs found in sweep/. "
              "Showing ablation results only.\n")

    groups = defaultdict(list)

    for d in sorted(SWEEP_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("EXP_"):
            continue
        parsed = parse_exp_dirname(d.name)
        if parsed is None:
            continue

        source, variant, stretch = parsed

        if args.source and source != args.source:
            continue
        if args.stretch:
            dir_stretch = stretch if stretch else "12"
            if dir_stretch != args.stretch:
                continue

        asrs = {}
        for prefix in MODEL_PREFIXES:
            asrs[f"{prefix}_asr"] = get_asr(d, prefix)

        row = {"name": d.name, "source": source, "variant": variant,
               "stretch": stretch or ""}
        row.update(asrs)
        row["mean_asr"] = _mean_asr(asrs)
        groups[(source, stretch or "")].append(row)

    if not groups and not baselines:
        print("No results found. Run experiments first.")
        sys.exit(1)

    all_keys = set(groups.keys())
    for src, st in baselines:
        if args.source and src != args.source:
            continue
        if args.stretch and st != args.stretch:
            continue
        all_keys.add((src, st))

    all_csv_rows = []

    for key in sorted(all_keys):
        source, stretch = key
        baseline_asrs = baselines.get(key, {})
        if not baseline_asrs and stretch == "":
            baseline_asrs = baselines.get((source, "12"), {})
        ablation_rows = groups.get(key, [])

        if not baseline_asrs and not ablation_rows:
            continue

        print_table(source, stretch, baseline_asrs, ablation_rows)

        if baseline_asrs:
            bl_row = {"name": f"{source}_baseline_s{stretch or '12'}",
                      "source": source, "variant": "baseline",
                      "stretch": stretch or "12"}
            bl_row.update(baseline_asrs)
            bl_row["mean_asr"] = _mean_asr(baseline_asrs)
            all_csv_rows.append(bl_row)
        all_csv_rows.extend(ablation_rows)

    print(
        "\n\nInterpretation:\n"
        "  Shuffle << baseline  => VLMs detect temporal ORDER\n"
        "  Static  << baseline  => Frame DIVERSITY needed\n"
        "  Reverse ~  baseline  => Directionality does not matter\n"
        "  S13     << G4        => Temporal training helps"
    )

    if args.csv:
        suffix = ""
        if args.source:
            suffix += f"_{args.source}"
        if args.stretch:
            suffix += f"_{args.stretch}"
        csv_path = SWEEP_ROOT / f"temporal_ablation_summary{suffix}.csv"

        fieldnames = ["name", "source", "variant", "stretch"]
        for prefix in MODEL_PREFIXES:
            fieldnames.append(f"{prefix}_asr")
        fieldnames.append("mean_asr")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_csv_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nCSV written to {csv_path}")


if __name__ == "__main__":
    main()
