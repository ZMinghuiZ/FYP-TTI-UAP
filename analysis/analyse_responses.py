#!/usr/bin/env python3
"""
Analyse VLM responses for temporal language across experimental conditions.

Parses `model_answer` from eval CSVs and compares the frequency of
accident-related and temporal-event language between G4 (temporal UAP),
S13 (no temporal), and other conditions.

Works with both:
  - Existing yes/no eval CSVs (some VLMs give explanations beyond "yes"/"no")
  - Descriptive eval CSVs from eval_descriptive.py (open-ended responses)

Usage:
    # Analyse existing yes/no eval results
    python analyse_responses.py --sweep_dir sweep/

    # Analyse descriptive eval results
    python analyse_responses.py --sweep_dir sweep/EXP_descriptive/

    # Analyse specific CSV files
    python analyse_responses.py --csv_files path/to/results1.csv path/to/results2.csv

    # Save comparative summary as CSV
    python analyse_responses.py --sweep_dir sweep/ --output_csv response_analysis.csv
"""

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Keyword lists ────────────────────────────────────────────────────────
# "Abnormal temporal": words suggesting something unusual happened at a
# specific moment.  Generic step-by-step words (then/after/before) are
# excluded because they appear in ALL descriptive responses.
TEMPORAL_ABNORMAL_KEYWORDS = [
    "suddenly", "abruptly", "unexpectedly", "without warning",
    "at that moment", "at this point", "at that point",
    "begins to crash", "starts to collide", "starts to spin",
    "moment of impact", "point of collision",
]

# Accident/anomaly language -- only *affirmed* mentions count (see
# negation-aware counting below).
ACCIDENT_KEYWORDS = [
    "accident", "crash", "collision", "collide", "collides",
    "overturned", "flipped", "flips", "wreck", "smash", "smashes",
    "debris", "damage", "damaged", "broken", "shattered",
    "fire", "smoke", "explosion", "explode",
    "impact",
    "anomaly", "anomalies", "anomalous",
    "on its side",
]

# Broader event descriptors that signal abnormal activity.
EVENT_KEYWORDS = [
    "swerve", "swerves", "swerving",
    "rear-end", "head-on", "t-bone", "side-swipe",
    "rolls over", "rolled over", "rolling over",
    "spins out", "spinning out", "spin out",
    "loses control", "lost control", "losing control",
    "brakes hard", "braking hard", "slams on the brakes",
    "skid", "skids", "skidding", "hydroplane",
    "runs a red", "ran a red",
    "veers", "veering", "veered",
    "strikes", "struck",
    "wrong side", "wrong way",
    "lying on the road", "lying on the ground",
    "erratic",
    "reckless", "recklessly",
    "ran over", "runs over", "running over",
]

# Negation markers for discounting negated accident mentions.
NEGATION_MARKERS = [
    "no ", "not ", "don't ", "doesn't ", "cannot ", "can't ",
    "isn't ", "aren't ", "without ", "absence of ",
    "no sign of ", "no evidence of ", "no indication of ",
    "does not appear", "doesn't appear",
    "no accident", "no collision", "no crash",
    "no anomal", "nothing unusual", "nothing abnormal",
]

# Explicit "nothing happened" language.
BENIGN_KEYWORDS = [
    "normal", "routine", "uneventful", "smooth",
    "safe", "safely", "nothing unusual", "nothing abnormal",
    "no incident", "no event", "ordinary",
    "typical", "regular", "peaceful",
]


def _word_count(text):
    """Approximate word count."""
    return len(text.split())


def count_keywords(text, keywords):
    """Return (count, matched_keywords) for simple substring matching."""
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw in text_lower]
    return len(matched), matched


def count_affirmed_keywords(text, keywords):
    """Count keywords that are NOT preceded by a negation within 40 chars.

    For each keyword match, check the 40-character window before it for
    negation markers.  Only count the keyword if no negation is found.
    """
    text_lower = text.lower()
    affirmed = []
    negated = []
    for kw in keywords:
        idx = text_lower.find(kw)
        while idx != -1:
            window_start = max(0, idx - 40)
            window = text_lower[window_start:idx]
            is_negated = any(neg in window for neg in NEGATION_MARKERS)
            if is_negated:
                negated.append(kw)
            else:
                affirmed.append(kw)
            idx = text_lower.find(kw, idx + len(kw))
    return len(affirmed), affirmed, len(negated), negated


def analyse_csv(csv_path):
    """Parse a single eval CSV and return per-video analysis."""
    results = []
    try:
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                answer = row.get("model_answer", "")
                prediction = row.get("prediction", "")
                status = row.get("status", "")
                label = row.get("label", "")
                if not answer or answer.startswith("ERROR"):
                    continue

                wc = _word_count(answer)

                temporal_count, temporal_matched = count_keywords(
                    answer, TEMPORAL_ABNORMAL_KEYWORDS)

                acc_affirmed, acc_affirmed_kw, acc_negated, _ = \
                    count_affirmed_keywords(answer, ACCIDENT_KEYWORDS)

                event_count, event_matched = count_keywords(
                    answer, EVENT_KEYWORDS)

                benign_count, benign_matched = count_keywords(
                    answer, BENIGN_KEYWORDS)

                results.append({
                    "filename": row.get("filename", ""),
                    "prediction": prediction,
                    "status": status,
                    "label": label,
                    "answer": answer,
                    "answer_len": len(answer),
                    "word_count": wc,
                    "temporal_count": temporal_count,
                    "temporal_matched": temporal_matched,
                    "accident_affirmed": acc_affirmed,
                    "accident_affirmed_kw": acc_affirmed_kw,
                    "accident_negated": acc_negated,
                    "event_count": event_count,
                    "event_matched": event_matched,
                    "benign_count": benign_count,
                    "benign_matched": benign_matched,
                })
    except Exception as e:
        print(f"  [WARNING] Could not parse {csv_path}: {e}")
    return results


def find_csvs_in_sweep(sweep_dir, config_pattern=None):
    """Find eval CSVs grouped by config directory."""
    grouped = defaultdict(list)
    sweep = Path(sweep_dir)
    for csv_path in sorted(sweep.rglob("*_results_*.csv")):
        rel = csv_path.relative_to(sweep)
        config_name = str(rel.parts[0]) if len(rel.parts) > 1 else "root"
        if config_pattern and config_pattern not in config_name:
            continue
        grouped[config_name].append(csv_path)
    return grouped


def detect_model_from_filename(fname):
    """Infer VLM name from CSV filename."""
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


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _mann_whitney_u(a, b):
    """Minimal Mann-Whitney U test (no scipy dependency).

    Returns (U, p_approx).  For n >= 20, uses normal approximation.
    For smaller samples, returns p=NaN (inspect U manually).
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")

    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])

    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined))
             if combined[k][1] == 0)
    u1 = r1 - na * (na + 1) / 2
    u2 = na * nb - u1
    U = min(u1, u2)

    if na >= 20 and nb >= 20:
        mu = na * nb / 2
        sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
        if sigma == 0:
            return U, float("nan")
        z = abs(U - mu) / sigma
        p = math.erfc(z / math.sqrt(2))
        return U, p
    return U, float("nan")


def summarise_condition(name, all_results, verbose=False):
    """Print summary statistics for one experimental condition."""
    if not all_results:
        print(f"  {name}: no results found")
        return {}

    n = len(all_results)
    has_predictions = any(r["prediction"] for r in all_results)

    n_yes = sum(1 for r in all_results if r["prediction"] == "yes")
    n_no = sum(1 for r in all_results if r["prediction"] == "no")

    avg_words = _mean([r["word_count"] for r in all_results])
    avg_len = _mean([r["answer_len"] for r in all_results])

    avg_temporal = _mean([r["temporal_count"] for r in all_results])
    avg_accident = _mean([r["accident_affirmed"] for r in all_results])
    avg_negated = _mean([r["accident_negated"] for r in all_results])
    avg_event = _mean([r["event_count"] for r in all_results])
    avg_benign = _mean([r["benign_count"] for r in all_results])

    per_100w_temporal = [
        r["temporal_count"] / r["word_count"] * 100
        if r["word_count"] > 0 else 0 for r in all_results]
    per_100w_accident = [
        r["accident_affirmed"] / r["word_count"] * 100
        if r["word_count"] > 0 else 0 for r in all_results]
    per_100w_event = [
        r["event_count"] / r["word_count"] * 100
        if r["word_count"] > 0 else 0 for r in all_results]

    has_temporal = sum(1 for r in all_results if r["temporal_count"] > 0)
    has_accident = sum(1 for r in all_results if r["accident_affirmed"] > 0)
    has_event = sum(1 for r in all_results if r["event_count"] > 0)
    has_benign = sum(1 for r in all_results if r["benign_count"] > 0)

    asr = n_yes / n * 100 if (n > 0 and has_predictions) else float("nan")

    print(f"\n  {name} ({n} responses, avg {avg_words:.0f} words)")
    if has_predictions:
        print(f"    ASR: {asr:.1f}% ({n_yes} yes / {n_no} no)")
    print(f"    Temporal (abnormal): {avg_temporal:.2f}/resp  "
          f"({has_temporal}/{n} = {has_temporal/n:.0%})  "
          f"[{_mean(per_100w_temporal):.2f}/100w]")
    print(f"    Accident (affirmed): {avg_accident:.2f}/resp  "
          f"({has_accident}/{n} = {has_accident/n:.0%})  "
          f"[{_mean(per_100w_accident):.2f}/100w]")
    print(f"    Accident (negated):  {avg_negated:.2f}/resp  "
          f"(discounted)")
    print(f"    Event descriptors:   {avg_event:.2f}/resp  "
          f"({has_event}/{n} = {has_event/n:.0%})  "
          f"[{_mean(per_100w_event):.2f}/100w]")
    print(f"    Benign language:     {avg_benign:.2f}/resp  "
          f"({has_benign}/{n} = {has_benign/n:.0%})")

    all_acc_words = Counter()
    all_event_words = Counter()
    for r in all_results:
        for w in r["accident_affirmed_kw"]:
            all_acc_words[w] += 1
        for w in r["event_matched"]:
            all_event_words[w] += 1

    if all_acc_words:
        top = all_acc_words.most_common(5)
        print(f"    Top accident words: "
              f"{', '.join(f'{w}({c})' for w, c in top)}")
    if all_event_words:
        top = all_event_words.most_common(5)
        print(f"    Top event words:    "
              f"{', '.join(f'{w}({c})' for w, c in top)}")

    examples = sorted(all_results,
                       key=lambda r: r["accident_affirmed"] + r["event_count"],
                       reverse=True)
    examples = [r for r in examples
                if r["accident_affirmed"] + r["event_count"] > 0]
    if examples:
        print(f"    Example responses with accident/event language:")
        limit = 5 if verbose else 3
        for ex in examples[:limit]:
            snippet = ex["answer"][:300]
            pred = f"[{ex['prediction']}] " if ex["prediction"] else ""
            print(f"      {pred}{snippet}...")

    return {
        "name": name, "n": n, "asr": asr,
        "avg_words": avg_words, "avg_len": avg_len,
        "avg_temporal": avg_temporal,
        "avg_accident_affirmed": avg_accident,
        "avg_accident_negated": avg_negated,
        "avg_event": avg_event, "avg_benign": avg_benign,
        "per100w_temporal": _mean(per_100w_temporal),
        "per100w_accident": _mean(per_100w_accident),
        "per100w_event": _mean(per_100w_event),
        "has_temporal_frac": has_temporal / n if n > 0 else 0,
        "has_accident_frac": has_accident / n if n > 0 else 0,
        "has_event_frac": has_event / n if n > 0 else 0,
        "has_benign_frac": has_benign / n if n > 0 else 0,
        "_raw_accident_per_video": [r["accident_affirmed"] for r in all_results],
        "_raw_event_per_video": [r["event_count"] for r in all_results],
        "_raw_temporal_per_video": [r["temporal_count"] for r in all_results],
    }


def write_output_csv(summaries, output_path):
    """Save the comparative summary table to CSV."""
    if not summaries:
        return
    fieldnames = [
        "model", "name", "n", "asr", "avg_words", "avg_len",
        "avg_temporal", "avg_accident_affirmed", "avg_accident_negated",
        "avg_event", "avg_benign",
        "per100w_temporal", "per100w_accident", "per100w_event",
        "has_temporal_frac", "has_accident_frac", "has_event_frac",
        "has_benign_frac",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in fieldnames}
            for k in fieldnames:
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.4f}"
            writer.writerow(row)
    print(f"\nSummary table saved to: {output_path}")


def write_per_video_csv(by_model, output_path):
    """Save per-video keyword analysis to CSV for detailed inspection."""
    fieldnames = [
        "model", "config", "filename", "prediction",
        "word_count", "temporal_count",
        "accident_affirmed", "accident_negated", "event_count",
        "benign_count",
        "temporal_matched", "accident_affirmed_kw", "event_matched",
        "answer",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in sorted(by_model.keys()):
            for config_name in sorted(by_model[model].keys()):
                for r in by_model[model][config_name]:
                    writer.writerow({
                        "model": model,
                        "config": config_name,
                        "filename": r["filename"],
                        "prediction": r["prediction"],
                        "word_count": r["word_count"],
                        "temporal_count": r["temporal_count"],
                        "accident_affirmed": r["accident_affirmed"],
                        "accident_negated": r["accident_negated"],
                        "event_count": r["event_count"],
                        "benign_count": r["benign_count"],
                        "temporal_matched": "; ".join(r["temporal_matched"]),
                        "accident_affirmed_kw": "; ".join(r["accident_affirmed_kw"]),
                        "event_matched": "; ".join(r["event_matched"]),
                        "answer": r["answer"][:500],
                    })
    print(f"Per-video analysis saved to: {output_path}")


def main():
    p = argparse.ArgumentParser(
        description="Analyse VLM responses for temporal language")
    p.add_argument("--sweep_dir", type=str, default="sweep/",
                   help="Sweep directory containing config subdirs with CSVs")
    p.add_argument("--csv_files", type=str, nargs="+", default=None,
                   help="Specific CSV files to analyse")
    p.add_argument("--configs", type=str, nargs="+", default=None,
                   help="Config names to compare (e.g. G4_a4-near-s8-high "
                        "S13_no-temporal-small). Default: auto-detect.")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Save comparative summary table to this CSV path")
    p.add_argument("--output_details", type=str, default=None,
                   help="Save per-video keyword analysis to this CSV path")
    p.add_argument("--verbose", action="store_true",
                   help="Show more example responses")
    args = p.parse_args()

    print("=" * 70)
    print("VLM RESPONSE ANALYSIS: Temporal Language Comparison")
    print("=" * 70)

    if args.csv_files:
        all_results = []
        for csv_path in args.csv_files:
            results = analyse_csv(csv_path)
            all_results.extend(results)
            print(f"  Loaded {len(results)} responses from {csv_path}")
        summarise_condition("all", all_results, verbose=args.verbose)
        return

    grouped = find_csvs_in_sweep(args.sweep_dir)
    if not grouped:
        print(f"No eval CSVs (*_results_*.csv) found in {args.sweep_dir}")
        sys.exit(1)

    target_configs = args.configs
    if target_configs is None:
        target_configs = sorted(grouped.keys())

    print(f"\nFound {len(grouped)} config(s) with eval CSVs:")
    for cfg in target_configs:
        if cfg in grouped:
            print(f"  {cfg}: {len(grouped[cfg])} CSV(s)")

    by_model = defaultdict(lambda: defaultdict(list))

    for config_name in target_configs:
        if config_name not in grouped:
            print(f"  [SKIP] {config_name}: not found")
            continue

        for csv_path in grouped[config_name]:
            model = detect_model_from_filename(csv_path.name)
            results = analyse_csv(csv_path)
            by_model[model][config_name].extend(results)

    summaries = []
    for model in sorted(by_model.keys()):
        print(f"\n{'=' * 60}")
        print(f"  {model}")
        print(f"{'=' * 60}")
        for config_name in target_configs:
            if config_name in by_model[model]:
                s = summarise_condition(
                    config_name, by_model[model][config_name],
                    verbose=args.verbose)
                if s:
                    s["model"] = model
                    summaries.append(s)

    if len(summaries) >= 2:
        print(f"\n{'=' * 70}")
        print("COMPARATIVE SUMMARY")
        print(f"{'=' * 70}")
        print("  (Accident/Event counts are negation-aware; /100w = per 100 words)")
        print()
        header = (f"{'Config':<28s} {'Model':<12s} {'Words':>6s} "
                  f"{'Acc/r':>6s} {'Acc/100w':>9s} "
                  f"{'Evt/r':>6s} {'Evt/100w':>9s} "
                  f"{'%w/acc':>7s} {'Benign':>7s}")
        print(header)
        print("-" * len(header))
        for s in summaries:
            print(f"{s['name']:<28s} {s['model']:<12s} "
                  f"{s['avg_words']:>6.0f} "
                  f"{s['avg_accident_affirmed']:>6.2f} "
                  f"{s['per100w_accident']:>9.2f} "
                  f"{s['avg_event']:>6.2f} "
                  f"{s['per100w_event']:>9.2f} "
                  f"{s['has_accident_frac']:>6.0%} "
                  f"{s['has_benign_frac']:>6.0%}")

        # Statistical tests: compare G4 vs S13 and G4 vs clean per model
        print(f"\n{'=' * 70}")
        print("STATISTICAL TESTS (Mann-Whitney U)")
        print(f"{'=' * 70}")
        by_model_summaries = defaultdict(list)
        for s in summaries:
            by_model_summaries[s["model"]].append(s)

        for model, model_sums in sorted(by_model_summaries.items()):
            if len(model_sums) < 2:
                continue
            print(f"\n  {model}:")
            for i, s_a in enumerate(model_sums):
                for s_b in model_sums[i + 1:]:
                    for metric_name, key in [
                        ("Accident (affirmed)", "_raw_accident_per_video"),
                        ("Event descriptors", "_raw_event_per_video"),
                    ]:
                        a_vals = s_a.get(key, [])
                        b_vals = s_b.get(key, [])
                        if not a_vals or not b_vals:
                            continue
                        U, p = _mann_whitney_u(a_vals, b_vals)
                        p_str = f"p={p:.4f}" if p == p else "p=N/A (n<20)"
                        sig = ""
                        if p == p:
                            if p < 0.001:
                                sig = " ***"
                            elif p < 0.01:
                                sig = " **"
                            elif p < 0.05:
                                sig = " *"
                        mean_a = _mean(a_vals)
                        mean_b = _mean(b_vals)
                        print(f"    {s_a['name']} vs {s_b['name']}")
                        print(f"      {metric_name}: "
                              f"{mean_a:.2f} vs {mean_b:.2f}  "
                              f"U={U:.0f}  {p_str}{sig}")

    if args.output_csv:
        write_output_csv(summaries, args.output_csv)

    if args.output_details:
        write_per_video_csv(by_model, args.output_details)

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print("  If G4 has significantly MORE affirmed accident/event language")
    print("  than S13 and clean (p < 0.05):")
    print("    => VLMs perceive temporal events in G4 adversarial videos")
    print("  If differences are not significant:")
    print("    => The temporal UAP does not change what VLMs 'describe',")
    print("       only whether they answer 'yes' or 'no'")
    print()
    print("  Key metrics (negation-aware):")
    print("    Acc/r     = avg affirmed accident keywords per response")
    print("    Acc/100w  = same, normalized per 100 words")
    print("    Evt/r     = avg event descriptors per response")
    print("    %w/acc    = fraction of responses with any affirmed accident word")
    print("    Benign    = fraction of responses with benign/normal language")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
