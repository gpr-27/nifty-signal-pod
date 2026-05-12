"""
pod.py
======
Signal pod: loads the fine-tuned LoRA adapter on top of TinyLlama-1.1B (or
Phi-2) and generates a structured NIFTY options trading signal from a market
state dict.

Key design decisions:
  - 4-bit quantization (bitsandbytes NF4) for CPU inference
  - Conviction field: generated as free text, then parsed out with regex.
    It is NOT a softmax probability over the direction token.
    See CONVICTION_DESIGN_NOTES below.
  - Mandatory fallback: any JSON parse failure → NEUTRAL, conviction 0.0
  - Single public function: generate_signal(market_state, rag_episodes=None)

CONVICTION_DESIGN_NOTES
-----------------------
After fine-tuning, the model generates conviction as a text token sequence,
e.g. "0.67".  This value comes from the training data distribution — the model
has learned to associate certain market regimes with certain conviction ranges.
It is NOT a softmax probability over [CE, PE, NEUTRAL], because:

  1. softmax(direction_token) measures how confident the model is in its next
     token prediction at that specific position, not how often signals of that
     type are correct at inference time.
  2. The training data contained explicit numeric conviction labels; the model
     has learned to reproduce them contextually.
  3. To make conviction informative, we:
     (a) clamp it to [0.30, 0.80] during training (the raw labels had this range)
     (b) evaluate calibration post-hoc (slope of conviction vs. accuracy > 0)
     (c) use it as a downstream filter (orchestrator threshold at 0.40)
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = Path(__file__).parent / "adapters" / "nifty_signal_pod"

SYSTEM_PROMPT = (
    "You are a trading signal generator for NIFTY 50 options. "
    "Analyze the provided market state snapshot and generate a structured trading signal. "
    "Return ONLY valid JSON matching the required schema. "
    'Schema: {"direction": "CE"|"PE"|"NEUTRAL", "conviction": float 0.0-1.0, '
    '"horizon": "intraday"|"next_session", "signal_id": string, "generated_at": string}'
)

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.1        # near-deterministic for structured output
VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS = {"intraday", "next_session"}

# ──────────────────────────────────────────────────────────────────────────────
# Module-level singletons (lazy-loaded)
# ──────────────────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    logger.info("Loading tokenizer: %s", BASE_MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading base model (4-bit) …")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    if ADAPTER_PATH.exists():
        logger.info("Loading LoRA adapter from %s", ADAPTER_PATH)
        _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    else:
        logger.warning("Adapter not found at %s — using base model only", ADAPTER_PATH)
        _model = base

    _model.eval()
    logger.info("Model ready.")


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────
def _build_prompt(market_state: dict, rag_episodes: Optional[list] = None) -> str:
    ms_json = json.dumps(market_state, separators=(",", ":"))

    if rag_episodes:
        context_parts = []
        for ep in rag_episodes:
            context_parts.append(
                f"[Episode {ep['episode_id']} | {ep['regime']}] "
                f"{ep['summary']}  →  outcome: {ep['outcome']}"
            )
        rag_block = "\n\nHistorical context (similar episodes):\n" + "\n".join(context_parts)
    else:
        rag_block = ""

    # TinyLlama chat format
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}{rag_block}</s>\n"
        f"<|user|>\n{ms_json}</s>\n"
        "<|assistant|>\n"
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Output parsing
# ──────────────────────────────────────────────────────────────────────────────
def _extract_json(raw_text: str) -> Optional[dict]:
    """Extract first valid JSON object from model output."""
    # strip leading/trailing whitespace
    text = raw_text.strip()

    # find the first { … } block
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    # find the matching closing brace
    depth = 0
    for i, ch in enumerate(text[brace_start:], start=brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate_and_fix(signal: dict) -> tuple[dict, list[str]]:
    """
    Validate the parsed signal dict.  Minor fixable issues are corrected.
    Returns (fixed_signal, list_of_warnings).
    """
    warnings: list[str] = []

    direction = signal.get("direction", "")
    if direction not in VALID_DIRECTIONS:
        warnings.append(f"invalid direction '{direction}' → replaced with NEUTRAL")
        signal["direction"] = "NEUTRAL"

    horizon = signal.get("horizon", "")
    if horizon not in VALID_HORIZONS:
        warnings.append(f"invalid horizon '{horizon}' → replaced with 'intraday'")
        signal["horizon"] = "intraday"

    conviction = signal.get("conviction")
    if not isinstance(conviction, (int, float)):
        warnings.append(f"non-numeric conviction '{conviction}' → replaced with 0.0")
        signal["conviction"] = 0.0
    else:
        cv = float(conviction)
        if not (0.0 <= cv <= 1.0):
            warnings.append(f"conviction {cv} out of [0,1] → clamped")
            signal["conviction"] = max(0.0, min(1.0, cv))

    return signal, warnings


# ──────────────────────────────────────────────────────────────────────────────
# FALLBACK signal
# ──────────────────────────────────────────────────────────────────────────────
def _fallback_signal(reason: str) -> dict:
    return {
        "direction": "NEUTRAL",
        "conviction": 0.0,
        "horizon": "intraday",
        "signal_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, reason + datetime.utcnow().isoformat())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "_fallback": True,
        "_fallback_reason": reason,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def generate_signal(
    market_state: dict,
    rag_episodes: Optional[list] = None,
) -> dict:
    """
    Generate a NIFTY options trading signal from a market state snapshot.

    Args:
        market_state : dict with keys:
            nifty_spot, atm_iv, iv_skew_25d, pcr, adx_14,
            realized_vol_5d, vix_india, dte_nearest, moneyness_band
        rag_episodes : optional list of retrieved historical episodes
                       (output of retrieve.retrieve())

    Returns:
        Signal dict with keys: direction, conviction, horizon,
        signal_id, generated_at.
        On any failure, returns NEUTRAL with conviction 0.0.
    """
    _load_model()

    prompt = _build_prompt(market_state, rag_episodes)

    try:
        inputs = _tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id,
            )
        # decode only the new tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = _tokenizer.decode(new_ids, skip_special_tokens=True)
        logger.debug("Raw model output: %s", raw_text)

    except Exception as exc:
        logger.error("Model inference failed: %s", exc)
        return _fallback_signal(f"inference_exception: {exc}")

    # parse
    parsed = _extract_json(raw_text)
    if parsed is None:
        logger.warning("JSON parse failed. Raw: %s", raw_text[:200])
        fb = _fallback_signal("json_parse_failed")
        fb["_raw_output"] = raw_text[:500]
        return fb

    fixed, warnings = _validate_and_fix(parsed)
    for w in warnings:
        logger.warning("Signal fix: %s", w)

    # ensure signal_id and generated_at are present
    if not fixed.get("signal_id"):
        fixed["signal_id"] = str(uuid.uuid4())
    if not fixed.get("generated_at"):
        fixed["generated_at"] = datetime.now(timezone.utc).isoformat()

    return fixed


# ──────────────────────────────────────────────────────────────────────────────
# CLI quick-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    test_state = {
        "nifty_spot": 23100.0,
        "atm_iv": 14.5,
        "iv_skew_25d": 1.2,
        "pcr": 1.05,
        "adx_14": 25.0,
        "realized_vol_5d": 11.0,
        "vix_india": 16.0,
        "dte_nearest": 3,
        "moneyness_band": "ATM",
    }

    signal = generate_signal(test_state)
    print(json.dumps(signal, indent=2))
