"""
rag_experiment.py
=================
Two-condition ablation: runs the full orchestrator evaluation over the eval
window (days 31–60) once WITHOUT RAG context and once WITH RAG context.

Outputs:
  results/predictions_no_rag.jsonl
  results/predictions_with_rag.jsonl
  results/rag_ablation_report.json   ← side-by-side comparison

The script also examines whether RAG changes conviction scores and whether
those changes are directionally justified by the similarity of retrieved
episodes to the current market state.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from orchestrator import run_eval_batch
from eval_suite import evaluate, print_report, THRESHOLDS

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PARQUET_PATH = Path(__file__).parent.parent / "slm_intern_data" / "market_states.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# Conviction-change analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyse_conviction_change(
    no_rag: list[dict],
    with_rag: list[dict],
    market_df: pd.DataFrame,
) -> dict:
    """
    For each row, compute:
      - delta_conviction = conviction_with_rag - conviction_no_rag
      - whether the RAG episodes' outcomes matched the pod's final direction
        (i.e., was the retrieved context justifiably influencing conviction?)

    Returns a summary dict for the report.
    """
    deltas = []
    justified_increases = 0
    unjustified_increases = 0
    justified_decreases = 0
    unjustified_decreases = 0

    for nr, wr in zip(no_rag, with_rag):
        c_no = float(nr.get("conviction", 0.0))
        c_with = float(wr.get("conviction", 0.0))
        delta = c_with - c_no
        deltas.append(delta)

        direction = wr.get("direction", "NEUTRAL")
        if direction == "NEUTRAL":
            continue

        # Was retrieved context "aligned" with the final direction?
        # We check via _rag_episode_outcomes stored by the orchestrator (if any)
        # For the ablation we compare against actual return if available
        actual_return = wr.get("_actual_return", None)
        if actual_return is None:
            continue

        correct = (
            (direction == "CE" and actual_return > 0)
            or (direction == "PE" and actual_return < 0)
        )

        if delta > 0.02:     # RAG increased conviction
            if correct:
                justified_increases += 1
            else:
                unjustified_increases += 1
        elif delta < -0.02:  # RAG decreased conviction
            if not correct:
                justified_decreases += 1
            else:
                unjustified_decreases += 1

    arr = np.array(deltas)
    t_stat, p_val = stats.ttest_1samp(arr, 0.0)

    return {
        "mean_delta_conviction": round(float(arr.mean()), 4),
        "std_delta_conviction": round(float(arr.std()), 4),
        "pct_increased": round(float((arr > 0.02).mean()), 4),
        "pct_decreased": round(float((arr < -0.02).mean()), 4),
        "pct_unchanged": round(float((np.abs(arr) <= 0.02).mean()), 4),
        "t_test_vs_zero": {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_val), 4)},
        "rag_increased_conviction_justified": justified_increases,
        "rag_increased_conviction_unjustified": unjustified_increases,
        "rag_decreased_conviction_justified": justified_decreases,
        "rag_decreased_conviction_unjustified": unjustified_decreases,
        "interpretation": _interpret_conviction_change(
            float(arr.mean()), float(p_val),
            justified_increases, unjustified_increases,
        ),
    }


def _interpret_conviction_change(
    mean_delta: float, p_val: float, justified: int, unjustified: int
) -> str:
    if p_val >= 0.10:
        return (
            "RAG did not significantly alter conviction scores "
            f"(mean delta={mean_delta:+.4f}, p={p_val:.3f}). "
            "Context retrieval had negligible calibration effect."
        )
    direction = "increased" if mean_delta > 0 else "decreased"
    ratio = justified / (justified + unjustified) if (justified + unjustified) > 0 else 0
    quality = "directionally justified" if ratio > 0.55 else "not reliably justified"
    return (
        f"RAG significantly {direction} conviction (mean delta={mean_delta:+.4f}, p={p_val:.3f}). "
        f"Of direction changes driven by RAG, {ratio*100:.0f}% were {quality}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Directional accuracy comparison
# ──────────────────────────────────────────────────────────────────────────────
def compare_accuracy(no_rag: list[dict], with_rag: list[dict]) -> dict:
    def _accuracy(preds: list[dict]) -> tuple[float, int, int]:
        hits, total = 0, 0
        for p in preds:
            d = p.get("direction")
            ret = p.get("_actual_return")
            if d == "NEUTRAL" or ret is None:
                continue
            total += 1
            if (d == "CE" and ret > 0) or (d == "PE" and ret < 0):
                hits += 1
        return (hits / total if total else 0.0, hits, total)

    acc_nr, h_nr, n_nr = _accuracy(no_rag)
    acc_wr, h_wr, n_wr = _accuracy(with_rag)

    # McNemar-like: count where they differ
    both_correct = sum(
        1 for nr, wr in zip(no_rag, with_rag)
        if _is_correct(nr) and _is_correct(wr)
    )
    nr_only = sum(
        1 for nr, wr in zip(no_rag, with_rag)
        if _is_correct(nr) and not _is_correct(wr)
    )
    wr_only = sum(
        1 for nr, wr in zip(no_rag, with_rag)
        if not _is_correct(nr) and _is_correct(wr)
    )

    return {
        "no_rag": {"accuracy": round(acc_nr, 4), "correct": h_nr, "actionable": n_nr},
        "with_rag": {"accuracy": round(acc_wr, 4), "correct": h_wr, "actionable": n_wr},
        "delta_accuracy": round(acc_wr - acc_nr, 4),
        "both_correct": both_correct,
        "no_rag_only_correct": nr_only,
        "with_rag_only_correct": wr_only,
        "verdict": "RAG improved accuracy" if acc_wr > acc_nr else
                   "RAG hurt accuracy" if acc_wr < acc_nr else
                   "RAG had no effect on accuracy",
    }


def _is_correct(p: dict) -> bool:
    d = p.get("direction", "NEUTRAL")
    ret = p.get("_actual_return")
    if d == "NEUTRAL" or ret is None:
        return False
    return (d == "CE" and ret > 0) or (d == "PE" and ret < 0)


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────
def run_ablation():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(PARQUET_PATH)

    logger.info("─── Condition A: WITHOUT RAG ───")
    no_rag_preds = run_eval_batch(
        parquet_path=str(PARQUET_PATH),
        output_path=str(RESULTS_DIR / "predictions_no_rag.jsonl"),
        use_rag=False,
    )

    logger.info("─── Condition B: WITH RAG (k=3) ───")
    rag_preds = run_eval_batch(
        parquet_path=str(PARQUET_PATH),
        output_path=str(RESULTS_DIR / "predictions_with_rag.jsonl"),
        use_rag=True,
    )

    # ── Run eval suite on both conditions ────────────────────────────────────
    logger.info("Running eval suite on both conditions …")
    eval_no_rag = evaluate(no_rag_preds, df)
    eval_with_rag = evaluate(rag_preds, df)

    logger.info("\n=== NO RAG RESULTS ===")
    print_report(eval_no_rag)

    logger.info("\n=== WITH RAG RESULTS ===")
    print_report(eval_with_rag)

    # ── Conviction change analysis ────────────────────────────────────────────
    conv_analysis = analyse_conviction_change(no_rag_preds, rag_preds, df)
    acc_comparison = compare_accuracy(no_rag_preds, rag_preds)

    # ── Build full ablation report ────────────────────────────────────────────
    report = {
        "experiment": "RAG ablation — NIFTY signal pod",
        "conditions": {
            "no_rag": {
                "eval_summary": eval_no_rag["summary"],
                "pass_fail": eval_no_rag["pass_fail"],
                "walk_forward_windows": eval_no_rag["walk_forward_windows"],
            },
            "with_rag": {
                "eval_summary": eval_with_rag["summary"],
                "pass_fail": eval_with_rag["pass_fail"],
                "walk_forward_windows": eval_with_rag["walk_forward_windows"],
            },
        },
        "accuracy_comparison": acc_comparison,
        "conviction_change_analysis": conv_analysis,
        "conclusion": _write_conclusion(acc_comparison, conv_analysis),
    }

    out = RESULTS_DIR / "rag_ablation_report.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Ablation report written → %s", out)

    print("\n" + "=" * 70)
    print("RAG ABLATION SUMMARY")
    print("=" * 70)
    print(f"  No-RAG accuracy    : {acc_comparison['no_rag']['accuracy']*100:.1f}%")
    print(f"  With-RAG accuracy  : {acc_comparison['with_rag']['accuracy']*100:.1f}%")
    print(f"  Delta accuracy     : {acc_comparison['delta_accuracy']:+.4f}")
    print(f"  Conviction change  : {conv_analysis['mean_delta_conviction']:+.4f} (mean)")
    print(f"  Verdict            : {acc_comparison['verdict']}")
    print(f"  Interpretation     : {conv_analysis['interpretation']}")
    print("=" * 70)

    return report


def _write_conclusion(acc: dict, conv: dict) -> str:
    delta = acc["delta_accuracy"]
    if abs(delta) < 0.01:
        quality = "marginal"
        direction = "negligible"
    elif delta > 0:
        quality = "positive"
        direction = "improved"
    else:
        quality = "negative"
        direction = "degraded"

    return (
        f"RAG context had a {quality} effect on directional accuracy "
        f"(delta={delta:+.4f}). Signal quality {direction}. "
        f"Conviction: {conv['interpretation']}"
    )


if __name__ == "__main__":
    run_ablation()
