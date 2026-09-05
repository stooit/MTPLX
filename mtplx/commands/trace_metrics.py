"""Pure arithmetic for recorded decode economics; never estimates an AR baseline."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def mtp_economics(receipt: dict, ar_tok_s: float | None = None) -> dict:
    """Compare actual delivered tokens / full decode time with a matched AR run.

    For fixed depth D, q = accepted / drafted is UNCONDITIONAL across all
    proposed positions. Away from terminal/bonus boundaries, tokens/round is
    1 + D*q. Conditional per-position acceptance instead gives
    1 + p1 + p1*p2 + ...; these two definitions must not be confused.
    Production conclusions use actual delivered tokens, including overhead.
    """
    tokens = _number(receipt.get("completion_tokens"))
    elapsed = _number(receipt.get("decode_elapsed_s"))
    rounds = _number(receipt.get("verify_calls"))
    acc = receipt.get("accepted_by_depth") or []
    drafted = receipt.get("drafted_by_depth") or []
    accepted = sum(float(x) for x in acc)
    proposed = sum(float(x) for x in drafted)
    verify = _number(receipt.get("verify_time_s"))
    draft = _number(receipt.get("draft_time_s"))
    result: dict[str, Any] = {
        "acceptance_definition": "accepted draft tokens / all proposed draft tokens",
        "acceptance": accepted / proposed if proposed else None,
        "tokens_per_verify": tokens / rounds if tokens and rounds else None,
        "decode_ms_per_token": 1000 * elapsed / tokens if elapsed and tokens else None,
        "verify_ms_per_round": (
            1000 * verify / rounds if verify is not None and rounds else None
        ),
        "draft_ms_per_round": (
            1000 * draft / rounds if draft is not None and rounds else None
        ),
        "cycle_ms": 1000 * elapsed / rounds if elapsed and rounds else None,
        "ar_tok_s": ar_tok_s,
        "speedup_vs_ar": None,
        "break_even_acceptance": None,
        "status": "matched_ar_measurement_required",
    }
    if receipt.get("repetition_stop_triggered"):
        result.update(status="guard_stopped_invalid_quality_sample",
                      guard_reason=receipt.get("repetition_stop_reason"))
        return result
    ar = _number(ar_tok_s)
    if ar and tokens and elapsed:
        result["speedup_vs_ar"] = (tokens / elapsed) / ar
        result["status"] = "measured_comparison_requires_matched_workload"
        fixed = rounds and drafted and all(float(n) == rounds for n in drafted)
        if fixed and not receipt.get("context_copy_accepted_tokens"):
            # Do not clamp: >1 means even perfect acceptance cannot amortize
            # this observed cycle cost; <0 means the cycle is already cheaper.
            result["break_even_acceptance"] = (elapsed / rounds * ar - 1) / len(drafted)
    return result


def sample_intervals(events: list[dict]) -> list[dict]:
    """Difference cumulative counters; leave missing observations missing.

    A gap is an observation gap, not an invented sequence of zero-TPS samples.
    A negative delta is a counter reset and is never rendered as performance.
    """
    samples = sorted((e for e in events if e.get("ev") == "s"), key=lambda e: e.get("ts", 0))
    result = []
    for prev, cur in pairwise(samples):
        duration = float(cur["ts"]) - float(prev["ts"])
        if duration <= 0:
            continue
        row = {"start_s": prev["ts"], "end_s": cur["ts"], "duration_s": duration,
               "observation_gap": duration > 2.5, "context_tokens": cur.get("ctx")}
        for key in ("gen", "vc", "vt", "dt", "at", "ct", "rt", "st", "bt", "cct", "cv", "evc"):
            before, after = _number(prev.get(key)), _number(cur.get(key))
            row[key] = after - before if before is not None and after is not None and after >= before else None
        for key in ("acc", "drf"):
            before, after = prev.get(key), cur.get(key)
            row[key] = ([b-a for a, b in zip(before, after)]
                        if isinstance(before, list) and isinstance(after, list)
                        and len(before) == len(after) and all(b >= a for a, b in zip(before, after)) else None)
        row["tok_s"] = row["gen"] / duration if row["gen"] is not None else None
        denominator = sum(row["drf"]) if row["drf"] is not None else 0
        row["acceptance"] = sum(row["acc"]) / denominator if row["acc"] is not None and denominator else None
        row["active_memory_bytes"] = cur.get("mem_active")
        row["cache_memory_bytes"] = cur.get("mem_cache")
        row["peak_memory_bytes"] = cur.get("mem_peak")
        row["verify_route"] = cur.get("route")
        result.append(row)
    return result
