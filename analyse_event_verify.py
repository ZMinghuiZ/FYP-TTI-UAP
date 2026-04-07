#!/usr/bin/env python3
"""
Analyse event verification results from eval_event_verify.py.

Computes false-positive rates (FPR) per event type per (model, condition),
runs Fisher's exact tests with Holm-Bonferroni correction, and performs a
temporal-vs-spatial interaction analysis to determine whether the temporal
UAP specifically increases dynamic-event hallucination.

Statistical design:
  - Primary comparison: G4 vs S13  (isolates temporal component)
  - Secondary: G4 vs clean, S13 vs clean  (sanity checks)
  - Correction: Holm-Bonferroni across all tests per model
  - Ambiguous parsing results are conservatively treated as "no"

Event classification (for interaction analysis):
  - Temporal (require cross-frame perception): loss_of_control, overturn
  - Spatial  (single-frame perceivable):       collision, fire_smoke
  - Artifact (direct perturbation detection):  visual_artifacts
  - Ambiguous:                                 pedestrian

Usage:
    python analyse_event_verify.py --sweep_dir sweep/EXP_event_verify/
    python analyse_event_verify.py --csv_files *.csv --output_csv results.csv
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

EVENT_COLUMNS = [
    "collision",
    "loss_of_control",
    "fire_smoke",
    "overturn",
    "pedestrian",
    "visual_artifacts",
]

EVENT_LABELS = {
    "collision": "Collision",
    "loss_of_control": "Loss of control",
    "fire_smoke": "Fire/smoke",
    "overturn": "Overturn",
    "pedestrian": "Pedestrian",
    "visual_artifacts": "Visual artifacts",
}

TEMPORAL_EVENTS = ["loss_of_control", "overturn"]
SPATIAL_EVENTS = ["fire_smoke"]
ARTIFACT_EVENT = ["visual_artifacts"]
AMBIGUOUS_EVENTS = ["collision", "pedestrian"]

CONDITION_ORDER_DEFAULT = ["G4_temporal", "S13_no_temporal", "clean"]


def _classify_condition(name):
    """Classify a condition label into (role, sort_key) for ordering.

    Roles: 'temporal' > 'no_temporal' > 'clean' > 'other'.
    """
    low = name.lower()
    if low.startswith("clean"):
        return ("clean", low)
    if "no_temporal" in low:
        return ("no_temporal", low)
    if "temporal" in low:
        return ("temporal", low)
    return ("other", low)


_ROLE_RANK = {"temporal": 0, "no_temporal": 1, "clean": 2, "other": 3}


def detect_condition_order(by_model):
    """Build a sorted condition list from loaded data."""
    all_conds = set()
    for cond_map in by_model.values():
        all_conds.update(cond_map.keys())
    return sorted(all_conds,
                  key=lambda c: (_ROLE_RANK.get(_classify_condition(c)[0], 9),
                                 _classify_condition(c)[1]))


def build_comparisons(condition_order):
    """Generate pairwise Fisher comparisons from detected conditions.

    Primary: temporal vs no_temporal (isolates temporal component).
    Secondary: each attack condition vs clean.
    """
    temporal = [c for c in condition_order
                if _classify_condition(c)[0] == "temporal"]
    no_temporal = [c for c in condition_order
                   if _classify_condition(c)[0] == "no_temporal"]
    clean = [c for c in condition_order
             if _classify_condition(c)[0] == "clean"]

    comparisons = []
    for t in temporal:
        for nt in no_temporal:
            tag = f"{t} vs {nt} [PRIMARY]"
            comparisons.append((t, nt, tag))
    for t in temporal:
        for cl in clean:
            comparisons.append((t, cl, f"{t} vs {cl}"))
    for nt in no_temporal:
        for cl in clean:
            comparisons.append((nt, cl, f"{nt} vs {cl}"))
    return comparisons


def detect_interaction_pair(condition_order):
    """Pick the best (temporal, non-temporal) pair for interaction analysis."""
    temporal = [c for c in condition_order
                if _classify_condition(c)[0] == "temporal"]
    no_temporal = [c for c in condition_order
                   if _classify_condition(c)[0] == "no_temporal"]
    if temporal and no_temporal:
        return temporal[0], no_temporal[0]
    return None, None

# ── Fisher's exact test (no scipy dependency) ─────────────────────────────

_MAX_N = 2000
_LOG_FACT = [0.0] * (_MAX_N + 1)
for _i in range(2, _MAX_N + 1):
    _LOG_FACT[_i] = _LOG_FACT[_i - 1] + math.log(_i)


def _log_choose(n, k):
    if k < 0 or k > n:
        return float("-inf")
    if n > _MAX_N:
        return (
            _lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1)
        )
    return _LOG_FACT[n] - _LOG_FACT[k] - _LOG_FACT[n - k]


def _lgamma(x):
    return math.lgamma(x)


def _hypergeom_log_pmf(k, N, K, n):
    """Log PMF of hypergeometric(k | N, K, n)."""
    return _log_choose(K, k) + _log_choose(N - K, n - k) - _log_choose(N, n)


def fishers_exact_two_sided(a, b, c, d):
    """Two-sided Fisher's exact test for a 2x2 table [[a,b],[c,d]].

    Returns (odds_ratio, p_value).  Sums probabilities of all tables
    no more probable than the observed table (the standard two-sided
    definition).
    """
    N = a + b + c + d
    K = a + c
    n = a + b

    if N == 0:
        return float("nan"), float("nan")

    log_p_obs = _hypergeom_log_pmf(a, N, K, n)

    lo = max(0, n + K - N)
    hi = min(n, K)

    p_value = 0.0
    for x in range(lo, hi + 1):
        log_p = _hypergeom_log_pmf(x, N, K, n)
        if log_p <= log_p_obs + 1e-10:
            p_value += math.exp(log_p)

    p_value = min(p_value, 1.0)

    denom = b * c
    odds_ratio = (a * d) / denom if denom > 0 else float("inf")

    return odds_ratio, p_value


# ── Holm-Bonferroni correction ────────────────────────────────────────────

def holm_bonferroni(tests):
    """Apply Holm-Bonferroni correction to a list of (label, p_value) tuples.

    Returns list of (label, original_p, adjusted_p, significant) tuples
    sorted by original p-value.
    """
    m = len(tests)
    if m == 0:
        return []

    sorted_tests = sorted(tests, key=lambda x: x[1])
    adjusted = []
    max_so_far = 0.0
    for i, (label, p) in enumerate(sorted_tests):
        adj_p = p * (m - i)
        adj_p = max(adj_p, max_so_far)
        adj_p = min(adj_p, 1.0)
        max_so_far = adj_p
        adjusted.append((label, p, adj_p, adj_p < 0.05))

    return adjusted


# ── Data loading ──────────────────────────────────────────────────────────

def _is_yes(value):
    """Conservative: only 'yes' counts; 'ambiguous' and 'error' -> False."""
    return str(value).strip().lower() == "yes"


def detect_model_from_filename(fname):
    fname = fname.lower()
    if "internvl" in fname:
        return "InternVL3"
    if "qwen" in fname:
        return "Qwen3-VL"
    if "llava" in fname:
        return "LLaVA-OV"
    if "videollama" in fname:
        return "VideoLLaMA3"
    return "Unknown"


def load_csv(csv_path):
    """Load one event-verification CSV, return list of row dicts."""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(row.get(col, "") == "error" for col in EVENT_COLUMNS):
                continue
            rows.append(row)
    return rows


def find_csvs_in_sweep(sweep_dir):
    """Find event-verify CSVs grouped by config directory."""
    grouped = defaultdict(list)
    sweep = Path(sweep_dir)
    for csv_path in sorted(sweep.rglob("*_results_*.csv")):
        rel = csv_path.relative_to(sweep)
        config_name = str(rel.parts[0]) if len(rel.parts) > 1 else "root"
        grouped[config_name].append(csv_path)
    return grouped


# ── Analysis functions ────────────────────────────────────────────────────

def compute_fpr_table(by_model):
    """Compute false-positive rate per (model, condition, event_type).

    Returns: {model: {condition: {event_col: (n_yes, n_total, fpr)}}}
    """
    table = {}
    for model, cond_data in by_model.items():
        table[model] = {}
        for condition, rows in cond_data.items():
            n = len(rows)
            if n == 0:
                continue
            table[model][condition] = {}
            for col in EVENT_COLUMNS:
                n_yes = sum(1 for r in rows if _is_yes(r.get(col, "")))
                fpr = n_yes / n
                table[model][condition][col] = (n_yes, n, fpr)
    return table


def compute_composite(rows):
    """Fraction of videos where VLM said 'yes' to any of the 5 event types
    (excluding visual_artifacts, which is an artifact-detection probe)."""
    if not rows:
        return 0, 0, 0.0
    halluc_cols = [c for c in EVENT_COLUMNS if c != "visual_artifacts"]
    n_any = sum(
        1 for r in rows
        if any(_is_yes(r.get(c, "")) for c in halluc_cols)
    )
    return n_any, len(rows), n_any / len(rows)


def run_pairwise_fisher(fpr_table, model, cond_a, cond_b):
    """Run Fisher's exact test for each event type between two conditions.

    Returns list of (event_col, a_yes, a_n, b_yes, b_n, OR, p).
    """
    results = []
    data_a = fpr_table.get(model, {}).get(cond_a, {})
    data_b = fpr_table.get(model, {}).get(cond_b, {})
    if not data_a or not data_b:
        return results

    for col in EVENT_COLUMNS:
        if col not in data_a or col not in data_b:
            continue
        a_yes, a_n, _ = data_a[col]
        b_yes, b_n, _ = data_b[col]
        a_no = a_n - a_yes
        b_no = b_n - b_yes
        odds, p = fishers_exact_two_sided(a_yes, a_no, b_yes, b_no)
        results.append((col, a_yes, a_n, b_yes, b_n, odds, p))
    return results


def interaction_analysis(fpr_table, model, cond_a="G4_temporal",
                         cond_b="S13_no_temporal"):
    """Compute FPR delta (cond_a - cond_b) for temporal vs spatial events.

    Returns dict with per-category mean deltas and per-event deltas.
    """
    data_a = fpr_table.get(model, {}).get(cond_a, {})
    data_b = fpr_table.get(model, {}).get(cond_b, {})
    if not data_a or not data_b:
        return None

    deltas = {}
    for col in EVENT_COLUMNS:
        if col in data_a and col in data_b:
            deltas[col] = data_a[col][2] - data_b[col][2]

    def _mean_delta(cols):
        vals = [deltas[c] for c in cols if c in deltas]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "per_event": deltas,
        "temporal_mean_delta": _mean_delta(TEMPORAL_EVENTS),
        "spatial_mean_delta": _mean_delta(SPATIAL_EVENTS),
        "artifact_delta": deltas.get("visual_artifacts", float("nan")),
    }


# ── Output helpers ────────────────────────────────────────────────────────

def _sig_stars(p):
    if p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def print_fpr_table(fpr_table, condition_order):
    """Print FPR table: rows = event types, columns = conditions per model."""
    for model in sorted(fpr_table.keys()):
        conditions = [c for c in condition_order
                      if c in fpr_table[model]]
        if not conditions:
            continue

        print(f"\n{'=' * 70}")
        print(f"  {model} — False Positive Rates")
        print(f"{'=' * 70}")

        header = f"  {'Event Type':<22s}"
        for cond in conditions:
            header += f"  {cond:>18s}"
        print(header)
        print("  " + "-" * (22 + 20 * len(conditions)))

        for col in EVENT_COLUMNS:
            row_str = f"  {EVENT_LABELS.get(col, col):<22s}"
            for cond in conditions:
                data = fpr_table[model].get(cond, {}).get(col)
                if data:
                    n_yes, n, fpr = data
                    row_str += f"  {fpr:>7.1%} ({n_yes:>3d}/{n:<3d})"
                else:
                    row_str += f"  {'—':>18s}"
            print(row_str)

        # Composite
        for cond in conditions:
            rows_key = f"_rows_{cond}"
            # Printed separately below
        print()


def print_fisher_results(all_tests, comparison_label):
    """Print Fisher's exact test results for one pairwise comparison."""
    if not all_tests:
        return
    print(f"\n  {comparison_label}:")
    print(f"    {'Event':<22s} {'FPR_A':>7s} {'FPR_B':>7s} "
          f"{'OR':>7s} {'p_raw':>8s} {'p_adj':>8s} {'sig':>4s}")
    print(f"    {'-' * 66}")
    for label, p_raw, p_adj, sig in all_tests:
        parts = label.split("|")
        if len(parts) == 5:
            evt, fpr_a, fpr_b, odds_str, _ = parts
            print(f"    {evt:<22s} {fpr_a:>7s} {fpr_b:>7s} "
                  f"{odds_str:>7s} {p_raw:>8.4f} {p_adj:>8.4f} "
                  f"{'*' * (3 if p_adj < 0.001 else 2 if p_adj < 0.01 else 1 if p_adj < 0.05 else 0):>4s}")


def write_output_csv(fpr_table, by_model, output_path, condition_order):
    """Write summary CSV with FPR per (model, condition, event_type)."""
    fieldnames = (
        ["model", "condition", "n"]
        + [f"fpr_{c}" for c in EVENT_COLUMNS]
        + [f"n_yes_{c}" for c in EVENT_COLUMNS]
        + ["composite_fpr", "composite_n_yes"]
    )
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in sorted(fpr_table.keys()):
            for cond in condition_order:
                data = fpr_table.get(model, {}).get(cond)
                if not data:
                    continue
                n = list(data.values())[0][1] if data else 0
                comp_yes, comp_n, comp_fpr = compute_composite(
                    by_model.get(model, {}).get(cond, []))
                row = {"model": model, "condition": cond, "n": n,
                       "composite_fpr": f"{comp_fpr:.4f}",
                       "composite_n_yes": comp_yes}
                for col in EVENT_COLUMNS:
                    if col in data:
                        n_yes, _, fpr = data[col]
                        row[f"fpr_{col}"] = f"{fpr:.4f}"
                        row[f"n_yes_{col}"] = n_yes
                    else:
                        row[f"fpr_{col}"] = ""
                        row[f"n_yes_{col}"] = ""
                writer.writerow(row)
    print(f"\nSummary CSV saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Analyse event verification results")
    p.add_argument("--sweep_dir", type=str, default=None,
                   help="Directory containing config subdirs with CSVs")
    p.add_argument("--csv_files", type=str, nargs="+", default=None,
                   help="Specific CSV files to analyse")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Save FPR summary table to CSV")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (default: 0.05)")
    args = p.parse_args()

    if not args.sweep_dir and not args.csv_files:
        args.sweep_dir = "sweep/EXP_event_verify/"

    print("=" * 70)
    print("EVENT VERIFICATION ANALYSIS (Pass 2)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    # by_model[model][condition] = [row_dicts]
    by_model = defaultdict(lambda: defaultdict(list))

    if args.csv_files:
        for csv_path in args.csv_files:
            model = detect_model_from_filename(Path(csv_path).name)
            rows = load_csv(csv_path)
            for row in rows:
                cond = row.get("label", "unknown")
                by_model[model][cond].append(row)
            print(f"  Loaded {len(rows)} rows from {csv_path} ({model})")
    else:
        grouped = find_csvs_in_sweep(args.sweep_dir)
        if not grouped:
            print(f"No CSVs found in {args.sweep_dir}")
            sys.exit(1)
        for config_name, csv_paths in sorted(grouped.items()):
            for csv_path in csv_paths:
                model = detect_model_from_filename(csv_path.name)
                rows = load_csv(csv_path)
                for row in rows:
                    cond = row.get("label", config_name)
                    by_model[model][cond].append(row)
                print(f"  {config_name}: {len(rows)} rows ({model})")

    # ── Detect condition order ────────────────────────────────────────
    condition_order = detect_condition_order(by_model)
    if not condition_order:
        print("No conditions found in data.")
        sys.exit(1)
    print(f"\n  Detected conditions: {condition_order}")

    # ── FPR table ─────────────────────────────────────────────────────
    fpr_table = compute_fpr_table(by_model)
    print_fpr_table(fpr_table, condition_order)

    # ── Composite hallucination score ─────────────────────────────────
    cond_col_w = max(len(c) for c in condition_order)
    print(f"\n{'=' * 70}")
    print("  Composite Hallucination Score (YES to any of events 1-5)")
    print(f"{'=' * 70}")
    for model in sorted(by_model.keys()):
        print(f"\n  {model}:")
        for cond in condition_order:
            rows = by_model.get(model, {}).get(cond, [])
            if not rows:
                continue
            n_any, n, fpr = compute_composite(rows)
            print(f"    {cond:<{cond_col_w}s}: {fpr:>7.1%} ({n_any}/{n})")

    # ── Fisher's exact tests ──────────────────────────────────────────
    comparisons = build_comparisons(condition_order)

    print(f"\n{'=' * 70}")
    print("  Fisher's Exact Tests (Holm-Bonferroni corrected)")
    print(f"{'=' * 70}")

    for model in sorted(fpr_table.keys()):
        print(f"\n  {'─' * 60}")
        print(f"  {model}")
        print(f"  {'─' * 60}")

        all_tests_for_model = []

        for cond_a, cond_b, comp_label in comparisons:
            fisher_results = run_pairwise_fisher(
                fpr_table, model, cond_a, cond_b)
            for col, a_yes, a_n, b_yes, b_n, odds, p in fisher_results:
                fpr_a = a_yes / a_n if a_n > 0 else 0
                fpr_b = b_yes / b_n if b_n > 0 else 0
                odds_str = f"{odds:.2f}" if odds != float("inf") else "inf"
                label = (f"{EVENT_LABELS.get(col, col)}|{fpr_a:.1%}|"
                         f"{fpr_b:.1%}|{odds_str}|{comp_label}")
                all_tests_for_model.append((label, p))

        if not all_tests_for_model:
            print("    No data for pairwise tests.")
            continue

        corrected = holm_bonferroni(all_tests_for_model)

        for cond_a, cond_b, comp_label in comparisons:
            relevant = [(lab, p_raw, p_adj, sig)
                        for lab, p_raw, p_adj, sig in corrected
                        if comp_label in lab]
            if relevant:
                print(f"\n  {comp_label}:")
                print(f"    {'Event':<22s} {'FPR_A':>7s} {'FPR_B':>7s} "
                      f"{'OR':>7s} {'p_raw':>8s} {'p_adj':>8s} {'':>4s}")
                print(f"    {'-' * 66}")
                for lab, p_raw, p_adj, sig in relevant:
                    parts = lab.split("|")
                    evt = parts[0]
                    fpr_a_s = parts[1] if len(parts) > 1 else "?"
                    fpr_b_s = parts[2] if len(parts) > 2 else "?"
                    odds_s = parts[3] if len(parts) > 3 else "?"
                    stars = _sig_stars(p_adj)
                    print(f"    {evt:<22s} {fpr_a_s:>7s} {fpr_b_s:>7s} "
                          f"{odds_s:>7s} {p_raw:>8.4f} {p_adj:>8.4f} "
                          f"{stars:>4s}")

    # ── Temporal vs Spatial interaction ────────────────────────────────
    ia_temporal, ia_baseline = detect_interaction_pair(condition_order)
    if ia_temporal and ia_baseline:
        print(f"\n{'=' * 70}")
        print(f"  Temporal vs Spatial Interaction Analysis "
              f"({ia_temporal} - {ia_baseline} FPR delta)")
        print(f"{'=' * 70}")
        print()
        print("  Event classification:")
        print(f"    Temporal  : {', '.join(TEMPORAL_EVENTS)}")
        print(f"    Spatial   : {', '.join(SPATIAL_EVENTS)}")
        print(f"    Artifact  : {', '.join(ARTIFACT_EVENT)}")
        print(f"    Ambiguous : {', '.join(AMBIGUOUS_EVENTS)}")
        print()
        print("  Hypothesis: temporal delta >> 0, spatial delta <= 0")
        print("  (Temporal UAP specifically increases dynamic-event "
              "hallucination)")
        print()

        for model in sorted(fpr_table.keys()):
            result = interaction_analysis(
                fpr_table, model, ia_temporal, ia_baseline)
            if result is None:
                continue

            print(f"  {model}:")
            print(f"    {'Event':<22s} "
                  f"{ia_temporal + ' FPR':>18s} "
                  f"{ia_baseline + ' FPR':>18s} "
                  f"{'Delta':>8s}  {'Category':<10s}")
            print(f"    {'-' * 80}")

            a_data = fpr_table[model].get(ia_temporal, {})
            b_data = fpr_table[model].get(ia_baseline, {})

            for col in EVENT_COLUMNS:
                delta = result["per_event"].get(col, float("nan"))
                a_fpr = (a_data.get(col, (0, 0, 0))[2]
                         if col in a_data else 0)
                b_fpr = (b_data.get(col, (0, 0, 0))[2]
                         if col in b_data else 0)

                if col in TEMPORAL_EVENTS:
                    cat = "TEMPORAL"
                elif col in SPATIAL_EVENTS:
                    cat = "SPATIAL"
                elif col in ARTIFACT_EVENT:
                    cat = "ARTIFACT"
                else:
                    cat = "AMBIGUOUS"

                delta_str = (f"{delta:>+7.1%}"
                             if delta == delta else "    N/A")
                print(f"    {EVENT_LABELS.get(col, col):<22s} "
                      f"{a_fpr:>17.1%} {b_fpr:>17.1%}  "
                      f"{delta_str}  {cat:<10s}")

            td = result["temporal_mean_delta"]
            sd = result["spatial_mean_delta"]
            ad = result["artifact_delta"]
            print(f"    {'─' * 80}")
            print(f"    Mean temporal delta : {td:>+7.1%}" if td == td
                  else "    Mean temporal delta :     N/A")
            print(f"    Mean spatial  delta : {sd:>+7.1%}" if sd == sd
                  else "    Mean spatial  delta :     N/A")
            print(f"    Artifact      delta : {ad:>+7.1%}" if ad == ad
                  else "    Artifact      delta :     N/A")

            if td == td and sd == sd:
                if td > 0.05 and sd <= 0.05:
                    print("    => SUPPORTED: temporal delta positive, "
                          "spatial delta near zero or negative")
                elif td > sd + 0.05:
                    print("    => PARTIAL: temporal delta exceeds "
                          "spatial delta")
                else:
                    print("    => NOT SUPPORTED: no clear interaction")
            print()

    # ── Interpretation ────────────────────────────────────────────────
    print(f"{'=' * 70}")
    print("INTERPRETATION GUIDE")
    print(f"{'=' * 70}")
    if ia_temporal and ia_baseline:
        print(f"  Primary test ({ia_temporal} vs {ia_baseline}) isolates "
              f"the temporal component.")
    print("  Temporal losses (lambda_traj, lambda_trans) redirect the")
    print("  gradient, trading spatial attack strength for temporal")
    print("  coherence (budget competition).")
    print()
    print("  Strong evidence for temporal observability:")
    print(f"    1. {', '.join(TEMPORAL_EVENTS)}: temporal >> baseline "
          f"(sig.)")
    print(f"    2. {', '.join(SPATIAL_EVENTS)}: temporal <= baseline")
    print("    3. This interaction shows VLMs process cross-frame info.")
    print()
    print("  Ambiguous handling: 'ambiguous' parse results counted as 'no'")
    print("  (conservative for false-positive rate estimation).")
    print(f"{'=' * 70}")

    # ── CSV output ────────────────────────────────────────────────────
    if args.output_csv:
        write_output_csv(fpr_table, by_model, args.output_csv,
                         condition_order)


if __name__ == "__main__":
    main()
