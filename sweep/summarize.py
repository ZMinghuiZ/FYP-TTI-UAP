#!/usr/bin/env python3
"""
Aggregate ASR results and diagnostic scores from all sweep runs.

Usage:
    python sweep/summarize.py                    # ASR table only
    python sweep/summarize.py --csv              # also writes sweep/summary.csv
    python sweep/summarize.py --diagnose         # include diagnostic ranking
    python sweep/summarize.py --diagnose --csv   # full output with CSV
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path


SWEEP_ROOT = Path(__file__).resolve().parent


# ── ASR parsing ───────────────────────────────────────────────────────────

def parse_summary_file(path):
    """Extract ASR from a *_summary_metrics.txt file."""
    text = path.read_text()
    match = re.search(r"ATTACK SUCCESS RATE \(ASR\):\s*([\d.]+)%", text)
    if match:
        return float(match.group(1))
    return None


def parse_csv_results(path):
    """Fallback: compute ASR from a results CSV (status column)."""
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


def get_asr(run_dir, model_prefix):
    """
    Try to find ASR for a model in a sweep run directory.
    Looks for summary_metrics.txt first, then falls back to CSV.
    """
    summary = run_dir / f"{model_prefix}_summary_metrics.txt"
    if summary.exists():
        asr = parse_summary_file(summary)
        if asr is not None:
            return asr

    for csv_path in sorted(run_dir.glob(f"{model_prefix}_results_*.csv")):
        asr = parse_csv_results(csv_path)
        if asr is not None:
            return asr

    return None


# ── Config label loading ──────────────────────────────────────────────────

KNOWN_CONFIGS = {
    "S1_wider-di":           "di_scale_low=0.5",
    "S2_wider-di-highprob":  "di_scale_low=0.5, di_prob=0.7",
    "S3_mid-resolution":     "image_size=384",
    "S4_mid-res-wider-di":   "image_size=384, di_scale_low=0.5",
    "S5_no-temporal":        "lambda_traj=0.0, lambda_trans=0.0",
    "S6_low-temporal":       "lambda_traj=0.1, lambda_trans=0.05",
    "S7_small-step":         "alpha=1/255",
    "S8_small-mod-temporal":  "alpha=1/255, lambda_traj=0.15, lambda_trans=0.1",
    "S9_smaller-step":       "alpha=0.75/255",
    "S10_small-wider-di":    "alpha=1/255, di_scale_low=0.7",
}


def load_config_label(run_dir):
    """Load a human-readable config summary for a run."""
    run_name = run_dir.name
    if run_name in KNOWN_CONFIGS:
        return KNOWN_CONFIGS[run_name]

    cfg_file = run_dir / "grid_config.txt"
    if cfg_file.exists():
        parts = []
        for line in cfg_file.read_text().strip().splitlines():
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key in ("alpha", "lambda_traj", "lambda_trans", "epochs"):
                parts.append(f"{key}={val}")
        return ", ".join(parts)

    return ""


# ── Diagnostic output parsing ─────────────────────────────────────────────

_RE_SHIFT = re.compile(
    r"Target similarity:.*shift=([+-]?\d+\.\d+)"
)
_RE_GAP = re.compile(
    r"Target-Negative gap:.*improvement=([+-]?\d+\.\d+)"
)
_RE_VARIATION = re.compile(
    r"Temporal variation:\s+(\d+\.\d+)"
)
_RE_CORR = re.compile(
    r"vs Accident template:.*correlation=([+-]?\d+\.\d+)"
)
_RE_MODEL_HEADER = re.compile(
    r"^--- (.+) ---$"
)


def parse_diagnose_output(run_dir):
    """
    Parse diagnose_output.txt and return per-model metrics.

    Returns dict with keys:
      shifts:       list of float (target sim shift per model)
      gap_impr:     list of float (target-neg gap improvement per model)
      variations:   list of float (temporal variation per model)
      correlations: list of float (template correlation per model, may be shorter)
    or None if the file doesn't exist / is unparseable.
    """
    diag_path = run_dir / "diagnose_output.txt"
    if not diag_path.exists():
        return None

    text = diag_path.read_text()

    shifts = [float(m.group(1)) for m in _RE_SHIFT.finditer(text)]
    gap_impr = [float(m.group(1)) for m in _RE_GAP.finditer(text)]
    variations = [float(m.group(1)) for m in _RE_VARIATION.finditer(text)]
    correlations = [float(m.group(1)) for m in _RE_CORR.finditer(text)]

    if not shifts:
        return None

    return {
        "shifts": shifts,
        "gap_impr": gap_impr,
        "variations": variations,
        "correlations": correlations,
    }


def compute_composite_score(diag):
    """
    Composite transferability score from diagnostic metrics.

    score = min(shifts) + 0.3 * mean(shifts) + 0.2 * mean(correlations)

    min(shifts) prioritises configs that work across all surrogates.
    """
    if diag is None:
        return None

    shifts = diag["shifts"]
    correlations = diag["correlations"]

    min_shift = min(shifts)
    mean_shift = sum(shifts) / len(shifts)

    mean_corr = 0.0
    if correlations:
        valid = [c for c in correlations if not (math.isnan(c) or math.isinf(c))]
        if valid:
            mean_corr = sum(valid) / len(valid)

    return min_shift + 0.3 * mean_shift + 0.2 * mean_corr


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarize sweep ASR results")
    parser.add_argument("--csv", action="store_true",
                        help="Write results to sweep/summary.csv")
    parser.add_argument("--diagnose", action="store_true",
                        help="Include diagnostic scores and ranking")
    args = parser.parse_args()

    run_dirs = sorted(
        d for d in SWEEP_ROOT.iterdir()
        if d.is_dir() and (d.name.startswith("S") or d.name.startswith("G"))
    )

    if not run_dirs:
        print(f"No sweep run directories found in {SWEEP_ROOT}")
        print("Expected directories like sweep/S1_*/, sweep/G1_*/, ...")
        sys.exit(1)

    rows = []

    MODEL_PREFIXES = ["internvl", "qwen", "llava_onevision", "videollama3"]
    MODEL_LABELS = ["InternVL", "Qwen3-VL", "LLaVA-OV", "VidLLaMA3"]

    baseline_row = {
        "run": "baseline (v13)",
        "changed": "(current settings)",
        "internvl_asr": 78.03,
        "qwen_asr": 10.40,
        "llava_onevision_asr": None,
        "videollama3_asr": None,
        "mean_asr": (78.03 + 10.40) / 2,
        "diag": None,
        "score": None,
    }
    rows.append(baseline_row)

    for d in run_dirs:
        asr_vals = {}
        for prefix in MODEL_PREFIXES:
            asr_vals[f"{prefix}_asr"] = get_asr(d, prefix)

        diag = parse_diagnose_output(d) if args.diagnose else None
        score = compute_composite_score(diag) if diag else None

        row = {
            "run": d.name,
            "changed": load_config_label(d),
            "diag": diag,
            "score": score,
        }
        row.update(asr_vals)

        available = [v for v in asr_vals.values() if v is not None]
        row["mean_asr"] = sum(available) / len(available) if available else None
        rows.append(row)

    # ── Print ASR table ───────────────────────────────────────────────────

    col_w_run = max(max(len(r["run"]) for r in rows), 4)
    col_w_chg = max(max(len(r["changed"]) for r in rows), 7)

    def fmt_asr(val):
        if val is None:
            return "  --  "
        return f"{val:6.2f}%"

    def fmt_score(val):
        if val is None:
            return "  --  "
        return f"{val:+.4f}"

    has_extra_models = any(
        r.get("llava_onevision_asr") is not None
        or r.get("videollama3_asr") is not None
        for r in rows
    )

    print()
    print("Hyperparameter Sweep -- ASR Comparison")

    header_parts = [
        f"{'Run':<{col_w_run}}  ",
        f"{'Changed':<{col_w_chg}}  ",
        f"{'InternVL':>9}  ",
        f"{'Qwen3-VL':>9}  ",
    ]
    if has_extra_models:
        header_parts.extend([
            f"{'LLaVA-OV':>9}  ",
            f"{'VidLLaMA3':>9}  ",
        ])
    header_parts.append(f"{'Mean ASR':>9}")
    if args.diagnose:
        header_parts.extend([
            f"  {'MinShift':>9}  ",
            f"{'MeanShift':>10}  ",
            f"{'MeanCorr':>9}  ",
            f"{'Score':>8}",
        ])

    header = "".join(header_parts)
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for r in rows:
        parts = [
            f"{r['run']:<{col_w_run}}  ",
            f"{r['changed']:<{col_w_chg}}  ",
            f"{fmt_asr(r.get('internvl_asr')):>9}  ",
            f"{fmt_asr(r.get('qwen_asr')):>9}  ",
        ]
        if has_extra_models:
            parts.extend([
                f"{fmt_asr(r.get('llava_onevision_asr')):>9}  ",
                f"{fmt_asr(r.get('videollama3_asr')):>9}  ",
            ])
        parts.append(f"{fmt_asr(r['mean_asr']):>9}")

        if args.diagnose:
            diag = r["diag"]
            if diag:
                min_s = min(diag["shifts"])
                mean_s = sum(diag["shifts"]) / len(diag["shifts"])
                valid_corr = [c for c in diag["correlations"]
                              if not (math.isnan(c) or math.isinf(c))]
                mean_c = sum(valid_corr) / len(valid_corr) if valid_corr else 0.0
                parts.extend([
                    f"  {min_s:+.4f}   ",
                    f"  {mean_s:+.4f}    ",
                    f"  {mean_c:+.4f}   ",
                    f"  {fmt_score(r['score']):>8}",
                ])
            else:
                parts.extend(["    --     ", "     --      ",
                              "    --     ", "      --  "])

        print("".join(parts))

    print(separator)

    # ── Best runs ─────────────────────────────────────────────────────────

    complete_asr = [r for r in rows[1:] if r["mean_asr"] is not None]
    if complete_asr:
        best = max(complete_asr, key=lambda r: r["mean_asr"])
        print(f"\nBest mean ASR: {best['run']} ({best['mean_asr']:.2f}%)")
        for prefix, label in zip(MODEL_PREFIXES, MODEL_LABELS):
            key = f"{prefix}_asr"
            with_val = [r for r in complete_asr if r.get(key) is not None]
            if with_val:
                b = max(with_val, key=lambda r: r[key])
                print(f"Best {label}: {b['run']} ({b[key]:.2f}%)")

    scored = []
    if args.diagnose:
        scored = [r for r in rows[1:] if r["score"] is not None]
        if scored:
            ranked = sorted(scored, key=lambda r: r["score"], reverse=True)
            print("\n--- Diagnostic Ranking (top candidates for VLM eval) ---")
            for i, r in enumerate(ranked, 1):
                diag = r["diag"]
                shifts_str = ", ".join(f"{s:+.4f}" for s in diag["shifts"])
                asr_parts = []
                for prefix, label in zip(MODEL_PREFIXES, MODEL_LABELS):
                    val = r.get(f"{prefix}_asr")
                    if val is not None:
                        asr_parts.append(f"{label}={val:.1f}%")
                asr_str = (f"ASR: {', '.join(asr_parts)}"
                           if asr_parts else "ASR: not evaluated")
                print(f"  #{i:2d}  {r['run']:<30s}  "
                      f"score={r['score']:+.4f}  "
                      f"shifts=[{shifts_str}]  "
                      f"{asr_str}")
        else:
            print("\nNo diagnostic data found. Run run_sweep_diagnose.sh first.")

    if not complete_asr and not scored:
        print("\nNo completed runs found yet.")

    # ── CSV output ────────────────────────────────────────────────────────

    if args.csv:
        csv_path = SWEEP_ROOT / "summary.csv"
        fieldnames = ["run", "changed"]
        for prefix in MODEL_PREFIXES:
            fieldnames.append(f"{prefix}_asr")
        fieldnames.append("mean_asr")
        if args.diagnose:
            fieldnames.extend([
                "min_shift", "mean_shift", "mean_corr", "score",
            ])

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                csv_row = {
                    "run": r["run"],
                    "changed": r["changed"],
                    "mean_asr": r["mean_asr"],
                }
                for prefix in MODEL_PREFIXES:
                    csv_row[f"{prefix}_asr"] = r.get(f"{prefix}_asr")
                if args.diagnose:
                    diag = r["diag"]
                    if diag:
                        valid_corr = [
                            c for c in diag["correlations"]
                            if not (math.isnan(c) or math.isinf(c))
                        ]
                        csv_row["min_shift"] = min(diag["shifts"])
                        csv_row["mean_shift"] = (
                            sum(diag["shifts"]) / len(diag["shifts"])
                        )
                        csv_row["mean_corr"] = (
                            sum(valid_corr) / len(valid_corr)
                            if valid_corr else 0.0
                        )
                        csv_row["score"] = r["score"]
                    else:
                        csv_row["min_shift"] = None
                        csv_row["mean_shift"] = None
                        csv_row["mean_corr"] = None
                        csv_row["score"] = None
                writer.writerow(csv_row)
        print(f"\nCSV written to {csv_path}")


if __name__ == "__main__":
    main()
