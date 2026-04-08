#!/usr/bin/env python3
"""
Summarize accident-positive evaluation CSVs from eval_ori.py.

This script is for evaluating model performance on real accident videos
(positive class), where:
  - prediction == "yes"  -> true positive
  - prediction == "no"   -> false negative

By default, ambiguous predictions are treated as misses (strict policy).
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize accident-video detection performance "
                    "(TPR/FNR) from *_results_*.csv files."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing eval CSVs (e.g., eval_results/accident_baseline).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_results_*.csv",
        help="CSV glob pattern to match inside results_dir (default: *_results_*.csv).",
    )
    parser.add_argument(
        "--ambiguous_policy",
        type=str,
        default="miss",
        choices=["miss", "exclude"],
        help=(
            "How to treat ambiguous/unknown predictions: "
            "'miss' counts them as false negatives, "
            "'exclude' removes them from denominator."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional output CSV for summary table.",
    )
    return parser.parse_args()


def detect_model_from_filename(filename):
    name = filename.lower()
    if "internvl" in name:
        return "InternVL3"
    if "qwen" in name:
        return "Qwen3-VL"
    if "llava" in name:
        return "LLaVA-OV"
    if "videollama" in name:
        return "VideoLLaMA3"
    return "Unknown"


def normalize_prediction(row):
    pred = str(row.get("prediction", "")).strip().lower()
    if pred in {"yes", "no", "ambiguous", "error"}:
        return pred

    status = str(row.get("status", "")).strip().lower()
    if status == "error":
        return "error"

    answer = str(row.get("model_answer", "")).strip().lower()
    if answer.startswith("error"):
        return "error"

    if pred.startswith("yes"):
        return "yes"
    if pred.startswith("no"):
        return "no"

    return "ambiguous"


def summarize_rows(rows, ambiguous_policy):
    counts = {
        "total_rows": 0,
        "errors": 0,
        "yes": 0,
        "no": 0,
        "ambiguous": 0,
        "denominator": 0,
        "tp": 0,
        "fn": 0,
        "tpr": float("nan"),
        "fnr": float("nan"),
    }

    for row in rows:
        counts["total_rows"] += 1
        pred = normalize_prediction(row)

        if pred == "error":
            counts["errors"] += 1
            continue
        if pred == "yes":
            counts["yes"] += 1
        elif pred == "no":
            counts["no"] += 1
        else:
            counts["ambiguous"] += 1

    if ambiguous_policy == "miss":
        denominator = counts["yes"] + counts["no"] + counts["ambiguous"]
        tp = counts["yes"]
        fn = counts["no"] + counts["ambiguous"]
    else:  # exclude
        denominator = counts["yes"] + counts["no"]
        tp = counts["yes"]
        fn = counts["no"]

    counts["denominator"] = denominator
    counts["tp"] = tp
    counts["fn"] = fn
    counts["tpr"] = (tp / denominator) if denominator > 0 else float("nan")
    counts["fnr"] = (fn / denominator) if denominator > 0 else float("nan")

    return counts


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt_pct(value):
    if value != value or math.isinf(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def print_summary(per_file, per_model, overall, ambiguous_policy):
    print("=" * 78)
    print("ACCIDENT VIDEO DETECTION SUMMARY (POSITIVE-CLASS EVAL)")
    print("=" * 78)
    print(f"Ambiguous policy: {ambiguous_policy}")
    print("TPR = yes / denominator, FNR = fn / denominator")
    print()

    print("- Per model -")
    header = (
        f"{'Model':<14} {'TPR':>9} {'FNR':>9} {'TP':>7} {'FN':>7} "
        f"{'Denom':>7} {'Ambig':>7} {'Err':>7}"
    )
    print(header)
    print("-" * len(header))
    for model in sorted(per_model.keys()):
        s = per_model[model]
        print(
            f"{model:<14} "
            f"{fmt_pct(s['tpr']):>9} {fmt_pct(s['fnr']):>9} "
            f"{s['tp']:>7} {s['fn']:>7} {s['denominator']:>7} "
            f"{s['ambiguous']:>7} {s['errors']:>7}"
        )

    print()
    print("- Overall (micro-average across loaded CSVs) -")
    print(
        f"TPR={fmt_pct(overall['tpr'])}, "
        f"FNR={fmt_pct(overall['fnr'])}, "
        f"TP={overall['tp']}, FN={overall['fn']}, "
        f"Denom={overall['denominator']}, "
        f"Ambiguous={overall['ambiguous']}, Errors={overall['errors']}"
    )

    print()
    print("- Per file -")
    for path_str, s in sorted(per_file.items()):
        print(
            f"{path_str}: "
            f"TPR={fmt_pct(s['tpr'])}, FNR={fmt_pct(s['fnr'])}, "
            f"TP={s['tp']}, FN={s['fn']}, Denom={s['denominator']}, "
            f"Ambig={s['ambiguous']}, Errors={s['errors']}"
        )


def to_rows_for_csv(per_model, overall, ambiguous_policy):
    rows = []
    for model in sorted(per_model.keys()):
        s = per_model[model]
        rows.append({
            "scope": "model",
            "name": model,
            "ambiguous_policy": ambiguous_policy,
            "tpr": f"{s['tpr']:.6f}" if s["tpr"] == s["tpr"] else "",
            "fnr": f"{s['fnr']:.6f}" if s["fnr"] == s["fnr"] else "",
            "tp": s["tp"],
            "fn": s["fn"],
            "denominator": s["denominator"],
            "ambiguous": s["ambiguous"],
            "errors": s["errors"],
            "yes": s["yes"],
            "no": s["no"],
            "total_rows": s["total_rows"],
        })

    rows.append({
        "scope": "overall",
        "name": "all_models",
        "ambiguous_policy": ambiguous_policy,
        "tpr": f"{overall['tpr']:.6f}" if overall["tpr"] == overall["tpr"] else "",
        "fnr": f"{overall['fnr']:.6f}" if overall["fnr"] == overall["fnr"] else "",
        "tp": overall["tp"],
        "fn": overall["fn"],
        "denominator": overall["denominator"],
        "ambiguous": overall["ambiguous"],
        "errors": overall["errors"],
        "yes": overall["yes"],
        "no": overall["no"],
        "total_rows": overall["total_rows"],
    })
    return rows


def write_summary_csv(output_path, rows):
    fieldnames = [
        "scope", "name", "ambiguous_policy",
        "tpr", "fnr", "tp", "fn", "denominator",
        "ambiguous", "errors", "yes", "no", "total_rows",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary CSV to: {output_path}")


def sum_counts(items):
    keys = ["total_rows", "errors", "yes", "no", "ambiguous", "denominator", "tp", "fn"]
    out = {k: 0 for k in keys}
    for item in items:
        for k in keys:
            out[k] += item[k]
    out["tpr"] = (out["tp"] / out["denominator"]) if out["denominator"] > 0 else float("nan")
    out["fnr"] = (out["fn"] / out["denominator"]) if out["denominator"] > 0 else float("nan")
    return out


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"ERROR: results_dir does not exist or is not a directory: {results_dir}")

    csv_paths = sorted(results_dir.rglob(args.glob))
    if not csv_paths:
        raise SystemExit(
            f"ERROR: No CSV files found under {results_dir} with pattern '{args.glob}'."
        )

    per_file = {}
    rows_by_model = defaultdict(list)

    for csv_path in csv_paths:
        rows = load_rows(csv_path)
        summary = summarize_rows(rows, args.ambiguous_policy)
        per_file[str(csv_path)] = summary

        model = detect_model_from_filename(csv_path.name)
        rows_by_model[model].extend(rows)

    per_model = {}
    for model, rows in rows_by_model.items():
        per_model[model] = summarize_rows(rows, args.ambiguous_policy)

    overall = sum_counts(list(per_model.values()))

    print_summary(per_file, per_model, overall, args.ambiguous_policy)

    if args.output_csv:
        rows_for_csv = to_rows_for_csv(per_model, overall, args.ambiguous_policy)
        write_summary_csv(args.output_csv, rows_for_csv)


if __name__ == "__main__":
    main()
