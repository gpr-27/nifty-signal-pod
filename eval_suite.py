"""
eval_suite.py
=============
Walk-forward evaluation suite for the NIFTY signal pod.

COMMIT THIS FILE BEFORE THE FIRST KAGGLE TRAINING RUN.

Pre-committed thresholds (set before seeing results):
  - Directional accuracy (non-NEUTRAL signals only) >= 52% = PASS (per window)
  - Schema pass rate  >= 99%  = PASS  (100% is target)
  - Orchestrator suppression rate <= 60% total (ADX < 20 suppression)
  - Low-conviction downgrade rate: report only, no hard threshold
  - Parse failure rate  < 1%  = PASS
  - High-VIX window accuracy  >= 48%  = PASS  (lower bar — harder regime)

Usage:
    python eval_suite.py --predictions results/predictions.jsonl
                         --market_states ../slm_intern_data/market_states.parquet
                         --output results/eval_report.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# Pre-committed pass/fail thresholds
# ──────────────────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "directional_accuracy_min": 0.52,      # per 5-day window, non-NEUTRAL predictions
    "schema_pass_rate_min": 0.99,
    "parse_failure_rate_max": 0.01,
    "suppression_rate_max": 0.60,          # total orchestrator suppress rate
    "high_vix_accuracy_min": 0.48,
    "high_vix_threshold": 18.0,            # India VIX >= 18 = high-VIX regime
}


# ──────────────────────────────────────────────────────────────────────────────
# Schema validation
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_KEYS = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS = {"intraday", "next_session"}


def validate_schema(signal: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(signal.keys())
    if missing:
        errors.append(f"missing keys: {missing}")
    direction = signal.get("direction")
    if direction not in VALID_DIRECTIONS:
        errors.append(f"invalid direction: {direction!r}")
    horizon = signal.get("horizon")
    if horizon not in VALID_HORIZONS:
        errors.append(f"invalid horizon: {horizon!r}")
    conviction = signal.get("conviction")
    if not isinstance(conviction, (int, float)):
        errors.append(f"conviction not a number: {conviction!r}")
    elif not (0.0 <= float(conviction) <= 1.0):
        errors.append(f"conviction out of range: {conviction}")
    if not signal.get("signal_id"):
        errors.append("empty signal_id")
    if not signal.get("generated_at"):
        errors.append("empty generated_at")
    return len(errors) == 0, errors


# ──────────────────────────────────────────────────────────────────────────────
# Conviction calibration helpers
# ──────────────────────────────────────────────────────────────────────────────
def conviction_calibration(convictions: list[float], correct: list[bool], n_bins: int = 5) -> dict:
    """
    Calibration analysis: for each conviction bin, what fraction of predictions
    were correct?  We do NOT build a reliability diagram here (the brief
    explicitly warns against that as insufficient).  We report per-bin
    accuracy, and also the slope of a linear regression of conviction vs
    binary correctness — a slope > 0 means higher conviction correlates with
    better accuracy, which is the minimum bar for meaningful conviction.
    """
    if len(convictions) < 10:
        return {"calibration": "insufficient data", "slope": None, "p_value": None}

    c = np.array(convictions)
    y = np.array(correct, dtype=float)

    if np.unique(c).size <= 1:
        return {"calibration": "all conviction values identical — cannot regress", "slope": None, "p_value": None}

    # linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(c, y)

    # bin-level accuracy
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (c >= lo) & (c < hi)
        if mask.sum() > 0:
            bin_stats.append({
                "bin": f"[{lo:.1f}, {hi:.1f})",
                "n": int(mask.sum()),
                "accuracy": float(y[mask].mean()),
                "mean_conviction": float(c[mask].mean()),
            })

    return {
        "bins": bin_stats,
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_value**2), 4),
        "p_value": round(float(p_value), 4),
        "interpretation": (
            "conviction is informative (slope > 0, p < 0.10)"
            if slope > 0 and p_value < 0.10
            else "conviction is NOT reliably informative"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Parquet schema helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_day_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a 'day' column (1-based integer rank of trading date).
    The parquet ships without a 'day' column; we derive it from 'timestamp'.
    Day 1 = first unique trading date, day N = last.
    """
    if "day" in df.columns:
        return df
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame has neither 'day' nor 'timestamp' column")
    df = df.copy()
    df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
    unique_dates = sorted(df["_date"].unique())
    date_to_day = {d: i + 1 for i, d in enumerate(unique_dates)}
    df["day"] = df["_date"].map(date_to_day)
    df = df.drop(columns=["_date"])
    return df


def _actual_direction(row: Any) -> str | None:
    """
    Derive actual market direction from a parquet row.
    Prefers 'next_return' (float) if present and finite; falls back to
    'label' column (CE / PE / NEUTRAL string), which is the manually
    assigned correct signal.
    Returns 'CE', 'PE', or None (when direction is unknown or NEUTRAL).
    """
    ret = row.get("next_return", None)
    # Guard against pandas NaN (which is not None but is not a real value)
    if ret is not None and ret == ret:  # NaN != NaN
        ret = float(ret)
        if ret > 0:
            return "CE"
        if ret < 0:
            return "PE"
        return None  # exactly zero → skip
    label = row.get("label", None)
    if label in ("CE", "PE"):
        return label
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def evaluate(
    predictions: list[dict],
    market_states: pd.DataFrame,
    window_size: int = 5,
) -> dict:
    """
    Run the complete eval suite.

    predictions : list of orchestrator output dicts, one per eval-window row.
                  Each must contain the orchestrator output plus a 'row_idx'
                  field referencing the market_states row it corresponds to.

    market_states : the full parquet dataframe (all 60 days).

    Returns a nested dict of all metrics.
    """
    # Ensure 'day' column exists (derived from 'timestamp' if absent)
    market_states = _ensure_day_column(market_states)

    # Identify eval window rows (day 31–60)
    eval_df = market_states[market_states["day"] >= 31].reset_index(drop=True)

    if len(predictions) != len(eval_df):
        raise ValueError(
            f"predictions length {len(predictions)} != eval rows {len(eval_df)}"
        )

    # ── Schema pass rate ─────────────────────────────────────────────────────
    schema_results = [validate_schema(p) for p in predictions]
    schema_pass = sum(1 for ok, _ in schema_results if ok)
    schema_pass_rate = schema_pass / len(predictions)

    # ── Parse failure rate ───────────────────────────────────────────────────
    parse_failures = sum(
        1 for p in predictions if p.get("_parse_failed", False)
    )
    parse_failure_rate = parse_failures / len(predictions)

    # ── Orchestrator breakdown ───────────────────────────────────────────────
    suppressed = sum(
        1 for p in predictions if p.get("_reason_code") == "ADX_BELOW_20"
    )
    downgraded = sum(
        1 for p in predictions
        if p.get("_reason_code") == "LOW_CONVICTION"
    )
    suppression_rate = suppressed / len(predictions)

    # ── Walk-forward directional accuracy (per 5-day window) ────────────────
    n_days = eval_df["day"].nunique()
    unique_days = sorted(eval_df["day"].unique())
    windows = [
        unique_days[i: i + window_size]
        for i in range(0, len(unique_days), window_size)
    ]

    window_results = []
    all_convictions: list[float] = []
    all_correct: list[bool] = []

    for win_days in windows:
        win_mask = eval_df["day"].isin(win_days)
        win_preds = [p for p, m in zip(predictions, win_mask) if m]
        win_rows = eval_df[win_mask].reset_index(drop=True)

        # Directional signals = non-NEUTRAL predictions that weren't suppressed
        actionable = [
            (p, r)
            for p, (_, r) in zip(win_preds, win_rows.iterrows())
            if p.get("direction") != "NEUTRAL"
        ]

        if not actionable:
            window_results.append({
                "days": win_days,
                "n_total": len(win_preds),
                "n_actionable": 0,
                "directional_accuracy": None,
                "ci_low": None,
                "ci_high": None,
                "suppression_rate": sum(
                    1 for p in win_preds if p.get("_reason_code") == "ADX_BELOW_20"
                ) / len(win_preds),
                "downgrade_rate": sum(
                    1 for p in win_preds if p.get("_reason_code") == "LOW_CONVICTION"
                ) / len(win_preds),
            })
            continue

        correct_count = 0
        scored_count = 0
        for pred, row in actionable:
            direction = pred.get("direction")
            actual_dir = _actual_direction(row)
            if actual_dir is None:
                continue
            scored_count += 1
            is_correct = direction == actual_dir
            if is_correct:
                correct_count += 1
            conv = pred.get("conviction")
            if isinstance(conv, (int, float)):
                all_convictions.append(float(conv))
                all_correct.append(is_correct)

        if scored_count == 0:
            window_results.append({
                "days": win_days,
                "n_total": len(win_preds),
                "n_actionable": len(actionable),
                "directional_accuracy": None,
                "ci_low": None,
                "ci_high": None,
                "suppression_rate": sum(
                    1 for p in win_preds if p.get("_reason_code") == "ADX_BELOW_20"
                ) / len(win_preds),
                "downgrade_rate": sum(
                    1 for p in win_preds if p.get("_reason_code") == "LOW_CONVICTION"
                ) / len(win_preds),
            })
            continue

        n_act = scored_count
        acc = correct_count / n_act
        ci_lo, ci_hi = _wilson_ci(correct_count, n_act)

        # Mean VIX for window
        vix_vals = win_rows.get("vix_india", pd.Series(dtype=float))
        mean_vix = float(vix_vals.mean()) if not vix_vals.empty else None

        window_results.append({
            "days": win_days,
            "n_total": len(win_preds),
            "n_actionable": n_act,
            "directional_accuracy": round(acc, 4),
            "ci_low": round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
            "pass": acc >= THRESHOLDS["directional_accuracy_min"],
            "suppression_rate": round(
                sum(1 for p in win_preds if p.get("_reason_code") == "ADX_BELOW_20") / len(win_preds), 4
            ),
            "downgrade_rate": round(
                sum(1 for p in win_preds if p.get("_reason_code") == "LOW_CONVICTION") / len(win_preds), 4
            ),
            "mean_vix": round(mean_vix, 2) if mean_vix else None,
        })

    # ── VIX regime slice ─────────────────────────────────────────────────────
    high_vix_preds, low_vix_preds = [], []
    for p, (_, row) in zip(predictions, eval_df.iterrows()):
        vix = row.get("vix_india", 0)
        if vix >= THRESHOLDS["high_vix_threshold"]:
            high_vix_preds.append(p)
        else:
            low_vix_preds.append(p)

    def _regime_accuracy(preds: list[dict], rows: pd.DataFrame) -> dict:
        act = [
            (p, r)
            for p, (_, r) in zip(preds, rows.iterrows())
            if p.get("direction") != "NEUTRAL"
        ]
        if not act:
            return {"n": 0, "accuracy": None}
        correct = 0
        scored = 0
        for p, r in act:
            actual_dir = _actual_direction(r)
            if actual_dir is None:
                continue
            scored += 1
            if p["direction"] == actual_dir:
                correct += 1
        if scored == 0:
            return {"n": len(act), "accuracy": None}
        acc = correct / scored
        ci_lo, ci_hi = _wilson_ci(correct, scored)
        return {
            "n": scored,
            "correct": correct,
            "accuracy": round(acc, 4),
            "ci_low": round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
        }

    high_vix_rows = eval_df[eval_df["vix_india"] >= THRESHOLDS["high_vix_threshold"]].reset_index(drop=True)
    low_vix_rows = eval_df[eval_df["vix_india"] < THRESHOLDS["high_vix_threshold"]].reset_index(drop=True)

    high_vix_result = _regime_accuracy(high_vix_preds, high_vix_rows)
    low_vix_result = _regime_accuracy(low_vix_preds, low_vix_rows)

    # ── Conviction calibration ───────────────────────────────────────────────
    calibration = conviction_calibration(all_convictions, all_correct)

    # ── Pass/fail summary ────────────────────────────────────────────────────
    overall_acc_vals = [
        w["directional_accuracy"] for w in window_results if w["directional_accuracy"] is not None
    ]
    overall_acc = float(np.mean(overall_acc_vals)) if overall_acc_vals else None

    pass_fail = {
        "schema_pass_rate": {
            "value": round(schema_pass_rate, 4),
            "threshold": THRESHOLDS["schema_pass_rate_min"],
            "pass": schema_pass_rate >= THRESHOLDS["schema_pass_rate_min"],
        },
        "parse_failure_rate": {
            "value": round(parse_failure_rate, 4),
            "threshold": THRESHOLDS["parse_failure_rate_max"],
            "pass": parse_failure_rate <= THRESHOLDS["parse_failure_rate_max"],
        },
        "overall_directional_accuracy": {
            "value": round(overall_acc, 4) if overall_acc else None,
            "threshold": THRESHOLDS["directional_accuracy_min"],
            "pass": (overall_acc >= THRESHOLDS["directional_accuracy_min"]) if overall_acc else False,
        },
        "suppression_rate": {
            "value": round(suppression_rate, 4),
            "threshold": THRESHOLDS["suppression_rate_max"],
            "pass": suppression_rate <= THRESHOLDS["suppression_rate_max"],
        },
        "high_vix_accuracy": {
            "value": high_vix_result.get("accuracy"),
            "threshold": THRESHOLDS["high_vix_accuracy_min"],
            "pass": (
                high_vix_result.get("accuracy") is not None
                and high_vix_result["accuracy"] >= THRESHOLDS["high_vix_accuracy_min"]
            ),
        },
    }

    return {
        "thresholds_committed": THRESHOLDS,
        "summary": {
            "total_eval_rows": len(predictions),
            "schema_pass_rate": round(schema_pass_rate, 4),
            "parse_failure_rate": round(parse_failure_rate, 4),
            "suppression_rate": round(suppression_rate, 4),
            "downgrade_rate": round(downgraded / len(predictions), 4),
            "overall_directional_accuracy": round(overall_acc, 4) if overall_acc else None,
        },
        "walk_forward_windows": window_results,
        "regime_slice": {
            "high_vix": high_vix_result,
            "low_vix": low_vix_result,
        },
        "conviction_calibration": calibration,
        "pass_fail": pass_fail,
    }


def print_report(result: dict) -> None:
    pf = result["pass_fail"]
    print()
    print("=" * 70)
    print("EVAL SUITE RESULTS")
    print("=" * 70)
    s = result["summary"]
    print(f"  Total eval rows          : {s['total_eval_rows']}")
    print(f"  Schema pass rate         : {s['schema_pass_rate']*100:.1f}%")
    print(f"  Parse failure rate       : {s['parse_failure_rate']*100:.2f}%")
    print(f"  Suppression rate (ADX)   : {s['suppression_rate']*100:.1f}%")
    print(f"  Downgrade rate (low conv): {s['downgrade_rate']*100:.1f}%")
    print(f"  Overall dir. accuracy    : "
          f"{s['overall_directional_accuracy']*100:.1f}%" if s['overall_directional_accuracy'] else "  N/A")
    print()
    print("── Walk-forward windows ─────────────────────────────────────────────")
    for w in result["walk_forward_windows"]:
        status = "✓" if w.get("pass") else "✗"
        acc_str = f"{w['directional_accuracy']*100:.1f}%" if w["directional_accuracy"] is not None else "  N/A"
        ci_str = (
            f" [{w['ci_low']*100:.1f}%, {w['ci_high']*100:.1f}%]"
            if w.get("ci_low") is not None else ""
        )
        print(f"  Days {w['days'][0]:2d}-{w['days'][-1]:2d} | acc={acc_str}{ci_str} | "
              f"suppress={w['suppression_rate']*100:.0f}% | "
              f"downgrade={w['downgrade_rate']*100:.0f}%  {status}")

    print()
    print("── Regime slice ─────────────────────────────────────────────────────")
    hv = result["regime_slice"]["high_vix"]
    lv = result["regime_slice"]["low_vix"]
    print(f"  High-VIX (≥{THRESHOLDS['high_vix_threshold']}) n={hv['n']}  acc={hv.get('accuracy', 'N/A')}")
    print(f"  Low-VIX             n={lv['n']}  acc={lv.get('accuracy', 'N/A')}")

    print()
    print("── Conviction calibration ───────────────────────────────────────────")
    cc = result["conviction_calibration"]
    print(f"  Slope vs accuracy        : {cc.get('slope')}  (p={cc.get('p_value')})")
    print(f"  Interpretation           : {cc.get('interpretation')}")

    print()
    print("── Pass / Fail vs pre-committed thresholds ──────────────────────────")
    for metric, res in pf.items():
        status = "PASS ✓" if res["pass"] else "FAIL ✗"
        val = f"{res['value']*100:.1f}%" if isinstance(res["value"], float) else str(res["value"])
        thr = f"{res['threshold']*100:.1f}%"
        print(f"  {metric:<40} {val:>8}  threshold={thr}  {status}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run eval suite on orchestrator predictions")
    parser.add_argument("--predictions", required=True, help="JSONL file of orchestrator outputs")
    parser.add_argument("--market_states", required=True, help="Path to market_states.parquet")
    parser.add_argument("--output", default="results/eval_report.json")
    args = parser.parse_args()

    preds = [json.loads(l) for l in Path(args.predictions).read_text().splitlines() if l.strip()]
    df = pd.read_parquet(args.market_states)

    result = evaluate(preds, df)
    print_report(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull report written → {out_path}")


if __name__ == "__main__":
    main()
