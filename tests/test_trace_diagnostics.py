import json
from argparse import Namespace

import pytest

from mtplx.commands.trace import _load_receipts, _match_receipt, _source_port
from mtplx.commands.trace_clients import load_pi_session
from mtplx.commands.trace_metrics import mtp_economics, sample_intervals


def test_economics_distinguishes_aggregate_acceptance_from_conditional_probability():
    receipt = {"completion_tokens": 250, "decode_elapsed_s": 4.0, "verify_calls": 100,
               "accepted_by_depth": [80, 50, 20], "drafted_by_depth": [100, 100, 100],
               "verify_time_s": 3, "draft_time_s": .8}
    result = mtp_economics(receipt, ar_tok_s=50)
    assert result["acceptance"] == .5
    assert result["tokens_per_verify"] == 2.5
    assert result["speedup_vs_ar"] == 1.25
    assert result["break_even_acceptance"] == pytest.approx(1/3)
    assert mtp_economics(receipt)["speedup_vs_ar"] is None
    # A copy route or mixed depths invalidates the fixed-depth threshold.
    assert mtp_economics({**receipt, "context_copy_accepted_tokens": 12}, 50)["break_even_acceptance"] is None
    assert mtp_economics({**receipt, "drafted_by_depth": [100, 80, 20]}, 50)["break_even_acceptance"] is None


def test_intervals_preserve_zero_and_missing_counters_and_expose_gaps():
    rows = sample_intervals([
        {"ev": "s", "ts": 1, "gen": 10, "vt": 0, "acc": [0], "drf": [0]},
        {"ev": "s", "ts": 2, "gen": 20, "vt": .4, "acc": [3], "drf": [5]},
        {"ev": "s", "ts": 9, "gen": 2, "vt": 0},
    ])
    assert rows[0]["vt"] == .4
    assert rows[0]["acceptance"] == .6
    assert rows[0]["dt"] is None
    assert rows[1]["observation_gap"] is True
    assert rows[1]["gen"] is None
    assert rows[1]["tok_s"] is None


def test_custom_logs_include_rotations_without_silently_using_latest_daemon(tmp_path):
    path = tmp_path / "receipts.jsonl"
    path.write_text('{"request_id":"new"}\n')
    (tmp_path / "receipts.jsonl.1").write_text('{"request_id":"old"}\n')
    assert [r["request_id"] for r in _load_receipts(0, path=str(path))] == ["old", "new"]
    assert _source_port(Namespace(port=None, request_log=str(path))) == 0


def test_pi_reader_follows_active_branch_and_keeps_correlation(tmp_path):
    rows = [{"type": "session", "id": "session", "cwd": str(tmp_path)},
            {"type": "message", "id": "u", "parentId": None,
             "message": {"role": "user", "timestamp": 1000, "content": "Build it"}},
            {"type": "message", "id": "abandoned", "parentId": "u",
             "message": {"role": "assistant", "timestamp": 2000, "content": []}},
            {"type": "message", "id": "current", "parentId": "u",
             "message": {"role": "assistant", "timestamp": 3000,
                         "content": [{"type": "thinking", "thinking": "inspect"}],
                         "usage": {"output": 20, "cacheRead": 12}}}]
    path = tmp_path / "pi.jsonl"
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    session, messages = load_pi_session(path)
    assert session["id"] == "session"
    assert [m["_id"] for m in messages] == ["u", "current"]
    assert messages[-1]["_parent_entry_id"] == "u"
    assert messages[-1]["_parts"] == [{"type": "reasoning", "text": "inspect"}]


def test_exact_pi_parent_beats_nearby_time_match_and_cannot_be_reused():
    message = {"_parent_entry_id": "right", "time": {"created": 1000}}
    receipts = [{"request_client_entry_id": "wrong", "logged_at_s": 1},
                {"request_client_entry_id": "right", "logged_at_s": 5}]
    used = set()
    assert _match_receipt(message, receipts, used) is receipts[1]
    assert used == {1}
