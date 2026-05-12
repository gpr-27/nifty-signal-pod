"""
orchestrator.py
===============
Wraps the signal pod and applies three suppression/filter rules in sequence:

  Rule 1 — ADX Gate:      if ADX_14 < 20 → suppress, return NEUTRAL
                          without calling the model at all.
  Rule 2 — Parse Gate:    if model output fails JSON parse → return NEUTRAL,
                          log raw output.
  Rule 3 — Conviction Gate: if conviction < 0.40 → downgrade direction to
                          NEUTRAL (conviction value is preserved).

Every decision is logged with:
  - reason_code (string)
  - triggering values (dict)
  - timestamp

The downstream pipeline reads ONLY the orchestrator output, never the raw pod
signal.  The orchestrator output schema is identical to the pod schema, with
two additional metadata fields: _reason_code, _trigger_values.

Usage:
    from orchestrator import Orchestrator
    orch = Orchestrator(use_rag=True)
    result = orch.run(market_state)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from pod import generate_signal

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Reason codes
# ──────────────────────────────────────────────────────────────────────────────
class ReasonCode:
    ADX_BELOW_20 = "ADX_BELOW_20"
    PARSE_FAILURE = "PARSE_FAILURE"
    LOW_CONVICTION = "LOW_CONVICTION"
    POD_SIGNAL_PASSED = "POD_SIGNAL_PASSED"


ADX_SUPPRESS_THRESHOLD = 20.0
CONVICTION_DOWNGRADE_THRESHOLD = 0.40


def _ensure_day_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'day' column (1-based integer trading-day rank) when absent.
    The parquet ships without a 'day' column; we derive it from 'timestamp'.
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

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# NEUTRAL helper
# ──────────────────────────────────────────────────────────────────────────────
def _neutral_signal(conviction: float = 0.0) -> dict:
    return {
        "direction": "NEUTRAL",
        "conviction": conviction,
        "horizon": "intraday",
        "signal_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator class
# ──────────────────────────────────────────────────────────────────────────────
class Orchestrator:
    def __init__(self, use_rag: bool = False, rag_k: int = 3):
        self.use_rag = use_rag
        self.rag_k = rag_k
        self._retrieve = None
        if use_rag:
            self._load_retriever()
        self._decision_log: list[dict] = []

    def _load_retriever(self):
        import importlib.util, sys
        retrieve_path = Path(__file__).parent.parent / "slm_intern_data" / "retrieve.py"
        spec = importlib.util.spec_from_file_location("retrieve", retrieve_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._retrieve = mod.retrieve

    def _log(self, market_state: dict, reason_code: str, trigger: dict, output: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason_code": reason_code,
            "trigger_values": trigger,
            "market_state_snippet": {
                k: market_state.get(k)
                for k in ("adx_14", "vix_india", "dte_nearest")
            },
            "output": output,
        }
        self._decision_log.append(entry)
        logger.info(
            "ORCHESTRATOR | %s | trigger=%s | direction=%s | conviction=%s",
            reason_code,
            json.dumps(trigger),
            output.get("direction"),
            output.get("conviction"),
        )

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, market_state: dict) -> dict:
        """
        Process one market state snapshot and return the final orchestrator output.
        """
        adx = market_state.get("adx_14", 0.0)

        # ── Rule 1: ADX gate ──────────────────────────────────────────────────
        if adx < ADX_SUPPRESS_THRESHOLD:
            output = _neutral_signal(conviction=0.0)
            output["_reason_code"] = ReasonCode.ADX_BELOW_20
            output["_trigger_values"] = {"adx_14": adx, "threshold": ADX_SUPPRESS_THRESHOLD}
            self._log(market_state, ReasonCode.ADX_BELOW_20,
                      {"adx_14": adx}, output)
            return output

        # ── Retrieve RAG context if enabled ──────────────────────────────────
        rag_episodes = None
        if self.use_rag and self._retrieve is not None:
            try:
                rag_episodes = self._retrieve(market_state, k=self.rag_k)
            except Exception as exc:
                logger.warning("RAG retrieval failed: %s — proceeding without context", exc)
                rag_episodes = None

        # ── Call pod ──────────────────────────────────────────────────────────
        pod_signal = generate_signal(market_state, rag_episodes=rag_episodes)

        # ── Rule 2: Parse failure gate ────────────────────────────────────────
        if pod_signal.get("_fallback", False):
            raw = pod_signal.get("_raw_output", "")
            output = _neutral_signal(conviction=0.0)
            output["_reason_code"] = ReasonCode.PARSE_FAILURE
            output["_trigger_values"] = {
                "fallback_reason": pod_signal.get("_fallback_reason", "unknown"),
                "raw_output_preview": raw[:200],
            }
            self._log(market_state, ReasonCode.PARSE_FAILURE,
                      output["_trigger_values"], output)
            return output

        # ── Rule 3: Conviction gate ───────────────────────────────────────────
        conviction = float(pod_signal.get("conviction", 0.0))
        if conviction < CONVICTION_DOWNGRADE_THRESHOLD:
            output = dict(pod_signal)
            output["direction"] = "NEUTRAL"
            output["_reason_code"] = ReasonCode.LOW_CONVICTION
            output["_trigger_values"] = {
                "conviction": conviction,
                "threshold": CONVICTION_DOWNGRADE_THRESHOLD,
                "original_direction": pod_signal.get("direction"),
            }
            self._log(market_state, ReasonCode.LOW_CONVICTION,
                      output["_trigger_values"], output)
            return output

        # ── Signal passes all gates ───────────────────────────────────────────
        output = dict(pod_signal)
        output["_reason_code"] = ReasonCode.POD_SIGNAL_PASSED
        output["_trigger_values"] = {
            "adx_14": adx,
            "conviction": conviction,
        }
        self._log(market_state, ReasonCode.POD_SIGNAL_PASSED,
                  output["_trigger_values"], output)
        return output

    def flush_log(self, path: Optional[Path] = None) -> None:
        """Write decision log to JSONL file."""
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            path = LOG_DIR / f"orchestrator_{ts}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self._decision_log:
                f.write(json.dumps(entry) + "\n")
        logger.info("Decision log flushed to %s (%d entries)", path, len(self._decision_log))

    def get_stats(self) -> dict:
        """Aggregate reason-code breakdown over all logged decisions."""
        from collections import Counter
        codes = Counter(e["reason_code"] for e in self._decision_log)
        total = len(self._decision_log)
        return {
            "total": total,
            "by_reason_code": dict(codes),
            "suppression_rate": codes[ReasonCode.ADX_BELOW_20] / total if total else 0,
            "parse_failure_rate": codes[ReasonCode.PARSE_FAILURE] / total if total else 0,
            "downgrade_rate": codes[ReasonCode.LOW_CONVICTION] / total if total else 0,
            "pass_rate": codes[ReasonCode.POD_SIGNAL_PASSED] / total if total else 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Batch evaluation runner
# ──────────────────────────────────────────────────────────────────────────────
def run_eval_batch(
    parquet_path: str,
    output_path: str,
    use_rag: bool = False,
    day_start: int = 31,
    day_end: int = 60,
) -> list[dict]:
    """
    Run the orchestrator over the eval window (days 31–60) and write
    predictions to a JSONL file.
    """
    df = pd.read_parquet(parquet_path)
    df = _ensure_day_column(df)
    eval_df = df[(df["day"] >= day_start) & (df["day"] <= day_end)].reset_index(drop=True)

    orch = Orchestrator(use_rag=use_rag)
    outputs = []

    for i, row in eval_df.iterrows():
        market_state = {
            "nifty_spot": row["nifty_spot"],
            "atm_iv": row["atm_iv"],
            "iv_skew_25d": row["iv_skew_25d"],
            "pcr": row["pcr"],
            "adx_14": row["adx_14"],
            "realized_vol_5d": row["realized_vol_5d"],
            "vix_india": row["vix_india"],
            "dte_nearest": row["dte_nearest"],
            "moneyness_band": row["moneyness_band"],
        }
        signal = orch.run(market_state)
        signal["_row_idx"] = int(i)
        signal["_day"] = int(row["day"])
        # preserve actual next return if available
        if "next_return" in row:
            signal["_actual_return"] = float(row["next_return"])
        outputs.append(signal)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d / %d rows", i + 1, len(eval_df))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")

    orch.flush_log()
    stats = orch.get_stats()
    logger.info("Eval batch complete. Stats: %s", stats)

    stats_path = out_path.parent / "orchestrator_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run orchestrator over eval window")
    parser.add_argument("--parquet", default="../slm_intern_data/market_states.parquet")
    parser.add_argument("--output", default="results/predictions.jsonl")
    parser.add_argument("--rag", action="store_true", help="Enable RAG")
    parser.add_argument("--day-start", type=int, default=31)
    parser.add_argument("--day-end", type=int, default=60)
    args = parser.parse_args()

    run_eval_batch(
        parquet_path=args.parquet,
        output_path=args.output,
        use_rag=args.rag,
        day_start=args.day_start,
        day_end=args.day_end,
    )
