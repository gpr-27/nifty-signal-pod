"""
data_audit.py
=============
Systematic audit of finetune_instructions.jsonl before any training.
Run this script and commit the output to the repo before the first Kaggle run.

Findings are printed to stdout and a cleaned file is written:
  cleaned_finetune_instructions.jsonl
"""

import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "slm_intern_data"
RAW_FILE = DATA_DIR / "finetune_instructions.jsonl"
CLEANED_FILE = Path(__file__).parent / "cleaned_finetune_instructions.jsonl"

VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS = {"intraday", "next_session"}
VALID_MONEYNESS = {"ATM", "1pct_OTM", "2pct_OTM"}

# Eval window dates start at day 31.  From the parquet we know training is
# Oct 1 – Oct 30 2024 (30 trading days).  Eval is Oct 31 onwards.
TRAIN_CUTOFF_DATE = "2024-10-30"

STRING_TO_FLOAT = {
    "high": 0.75,
    "strong": 0.75,
    "high confidence": 0.75,
    "moderate confidence": 0.60,
    "moderate": 0.55,
    "low": 0.35,
    "weak": 0.30,
}


def _parse_conviction(raw) -> tuple[float | None, str]:
    """
    Returns (float_value, fix_description).
    Returns (None, description) if unfixable.
    """
    if isinstance(raw, (int, float)):
        v = float(raw)
        if 0.0 <= v <= 1.0:
            return v, "ok"
        return None, f"out-of-range float {v}"

    if isinstance(raw, str):
        s = raw.strip().lower()

        # pattern like "0.8 (high)"
        m = re.match(r"^([0-9.]+)\s*\(", s)
        if m:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v, f"extracted numeric prefix from '{raw}'"
            return None, f"numeric prefix out of range in '{raw}'"

        # pure numeric string
        try:
            v = float(s)
            if 0.0 <= v <= 1.0:
                return v, f"parsed numeric string '{raw}'"
            return None, f"numeric string out of range in '{raw}'"
        except ValueError:
            pass

        # known verbal labels
        if s in STRING_TO_FLOAT:
            return STRING_TO_FLOAT[s], f"verbal label '{raw}' mapped to {STRING_TO_FLOAT[s]}"

        return None, f"unrecognised string conviction '{raw}'"

    return None, f"unexpected type {type(raw).__name__}"


def audit(verbose: bool = True) -> dict:
    raw_lines = RAW_FILE.read_text().splitlines()
    total = len(raw_lines)

    findings = {
        "total_rows": total,
        "verbal_conviction": [],          # conviction is a string label
        "conviction_numeric_prefix": [],  # "0.8 (high)" style
        "unfixable_conviction": [],       # cannot be salvaged
        "out_of_range_conviction": [],
        "invalid_direction": [],
        "invalid_horizon": [],
        "malformed_json_output": [],
        "atm_iv_floor": [],               # atm_iv == 10.0 (suspect floor/clamp)
        "duplicate_input": [],
        "eval_window_leak": [],           # training sample timestamped after cutoff
        "missing_fields": [],
    }

    input_hashes: dict[str, list[int]] = {}
    cleaned_rows = []

    for i, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue

        # --- parse outer JSONL ---
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            findings["malformed_json_output"].append(
                {"row": i, "issue": f"outer JSONL parse error: {e}"}
            )
            continue

        required_outer = {"instruction", "input", "output"}
        missing = required_outer - record.keys()
        if missing:
            findings["missing_fields"].append({"row": i, "missing": sorted(missing)})
            continue

        # --- parse input ---
        try:
            inp = json.loads(record["input"])
        except json.JSONDecodeError as e:
            findings["malformed_json_output"].append(
                {"row": i, "issue": f"input JSON parse error: {e}"}
            )
            continue

        # --- parse output ---
        try:
            out = json.loads(record["output"])
        except json.JSONDecodeError as e:
            findings["malformed_json_output"].append(
                {"row": i, "issue": f"output JSON parse error: {e}", "raw": record["output"]}
            )
            continue

        # --- schema checks on output ---
        conviction_raw = out.get("conviction")
        conviction_float, conviction_note = _parse_conviction(conviction_raw)

        if isinstance(conviction_raw, str):
            m = re.match(r"^([0-9.]+)\s*\(", conviction_raw.strip().lower())
            if m:
                findings["conviction_numeric_prefix"].append(
                    {"row": i, "raw": conviction_raw, "fix": conviction_note}
                )
            else:
                try:
                    float(conviction_raw)
                except ValueError:
                    findings["verbal_conviction"].append(
                        {"row": i, "raw": conviction_raw, "fix": conviction_note}
                    )

        if conviction_float is None:
            findings["unfixable_conviction"].append({"row": i, "raw": conviction_raw})
        elif isinstance(conviction_float, float) and not (0.0 <= conviction_float <= 1.0):
            findings["out_of_range_conviction"].append(
                {"row": i, "value": conviction_float}
            )

        direction = out.get("direction", "")
        if direction not in VALID_DIRECTIONS:
            findings["invalid_direction"].append({"row": i, "direction": direction})

        horizon = out.get("horizon", "")
        if horizon not in VALID_HORIZONS:
            findings["invalid_horizon"].append({"row": i, "horizon": horizon})

        # --- atm_iv floor check ---
        atm_iv = inp.get("atm_iv")
        if atm_iv == 10.0:
            findings["atm_iv_floor"].append({"row": i, "atm_iv": atm_iv})

        # --- eval window leak check ---
        generated_at = out.get("generated_at", "")
        if generated_at[:10] > TRAIN_CUTOFF_DATE:
            findings["eval_window_leak"].append(
                {"row": i, "generated_at": generated_at}
            )

        # --- duplicate input check ---
        input_key = record["input"]
        input_hashes.setdefault(input_key, []).append(i)

        # --- build clean row (drop eval-window leaks) ---
        is_eval_leak = generated_at[:10] > TRAIN_CUTOFF_DATE
        if (
            conviction_float is not None
            and direction in VALID_DIRECTIONS
            and horizon in VALID_HORIZONS
            and not is_eval_leak
        ):
            clean_out = dict(out)
            clean_out["conviction"] = round(conviction_float, 4)
            cleaned_record = dict(record)
            cleaned_record["output"] = json.dumps(clean_out)
            cleaned_rows.append(cleaned_record)

    for key, rows in input_hashes.items():
        if len(rows) > 1:
            findings["duplicate_input"].append({"rows": rows, "count": len(rows)})

    # ------------------------------------------------------------------ report
    if verbose:
        print("=" * 70)
        print("DATA AUDIT REPORT — finetune_instructions.jsonl")
        print("=" * 70)
        print(f"Total rows examined          : {total}")
        print()

        print("── Finding 1: Non-float conviction (verbal labels) ─────────────")
        print(f"  Affected rows              : {len(findings['verbal_conviction'])}")
        if findings["verbal_conviction"]:
            sample = findings["verbal_conviction"][:5]
            for s in sample:
                print(f"    row {s['row']:3d}  raw='{s['raw']}'  fix='{s['fix']}'")
            print(f"  Range of affected rows     : {findings['verbal_conviction'][0]['row']} – {findings['verbal_conviction'][-1]['row']}")
        print()

        print("── Finding 2: Mixed-format conviction ('0.8 (high)' style) ─────")
        print(f"  Affected rows              : {len(findings['conviction_numeric_prefix'])}")
        if findings["conviction_numeric_prefix"]:
            for s in findings["conviction_numeric_prefix"][:3]:
                print(f"    row {s['row']:3d}  raw='{s['raw']}'  fix='{s['fix']}'")
        print()

        print("── Finding 3: Unfixable conviction ──────────────────────────────")
        print(f"  Affected rows              : {len(findings['unfixable_conviction'])}")
        print()

        print("── Finding 4: atm_iv == 10.0 (apparent floor/clamp) ────────────")
        print(f"  Affected rows              : {len(findings['atm_iv_floor'])}")
        print("  Interpretation: Likely a data-pipeline floor applied when")
        print("  measured IV fell below 10%. Not a labelling error — these rows")
        print("  are kept but flagged. The model will learn this floor exists.")
        print()

        print("── Finding 5: Duplicate input vectors ───────────────────────────")
        dup_count = sum(len(d["rows"]) - 1 for d in findings["duplicate_input"])
        print(f"  Duplicate groups           : {len(findings['duplicate_input'])}")
        print(f"  Extra (redundant) rows     : {dup_count}")
        print()

        print("── Finding 6: Eval window leakage ───────────────────────────────")
        print(f"  Affected rows              : {len(findings['eval_window_leak'])}")
        if findings["eval_window_leak"]:
            for s in findings["eval_window_leak"][:3]:
                print(f"    row {s['row']:3d}  generated_at={s['generated_at']}")
        print()

        print("── Finding 7: Invalid direction / horizon ────────────────────────")
        print(f"  Invalid direction rows     : {len(findings['invalid_direction'])}")
        print(f"  Invalid horizon rows       : {len(findings['invalid_horizon'])}")
        print()

        print("── Finding 8: Malformed JSON in output field ────────────────────")
        print(f"  Affected rows              : {len(findings['malformed_json_output'])}")
        print()

        total_dropped = total - len(cleaned_rows)
        print("─" * 70)
        print(f"Rows retained (clean)        : {len(cleaned_rows)}")
        print(f"Rows dropped                 : {total_dropped}")
        print(f"Retention rate               : {len(cleaned_rows)/total*100:.1f}%")
        print()
        print("Decision rationale:")
        print("  Verbal conviction labels (Finding 1) are mapped to plausible")
        print("  numeric equivalents using a deterministic lexicon so signal is")
        print("  preserved rather than discarded. The mapping is documented above.")
        print("  Mixed-format ('0.8 (high)') rows use the numeric prefix only.")
        print("  Rows where conviction cannot be parsed at all are dropped —")
        print("  they would train the model to emit unparseable outputs.")
        print("  atm_iv floor rows are KEPT — the floor is a real market feature.")
        print("  Duplicate inputs are deduplicated (first occurrence kept).")
        print("  Eval-window leaks are dropped to prevent data contamination.")
        print("=" * 70)

    # ------------------------------------------------------------------ write
    CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)

    # deduplicate by input key, keep first clean occurrence
    seen_inputs: set[str] = set()
    deduped_rows = []
    for r in cleaned_rows:
        key = r["input"]
        if key not in seen_inputs:
            seen_inputs.add(key)
            deduped_rows.append(r)

    with CLEANED_FILE.open("w") as f:
        for r in deduped_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nCleaned file written → {CLEANED_FILE}")
    print(f"Final rows after dedup: {len(deduped_rows)}")

    findings["cleaned_count"] = len(deduped_rows)
    return findings


if __name__ == "__main__":
    audit(verbose=True)
