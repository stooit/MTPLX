"""The startup probe must test the producer the hardware will actually run."""

import pytest

from mtplx.kernels import qsa_indexer_select
from mtplx.models import qwen4_exp


@pytest.mark.parametrize("nax,consumer,mpp,expected", [
    (False, True, False, True),
    (False, False, False, False),
    (True, True, False, False),
    (True, True, True, True),
])
def test_auto_gate_keeps_portable_lane_and_rejects_broken_mpp(
    monkeypatch, nax, consumer, mpp, expected,
):
    monkeypatch.delenv("MTPLX_QSA_PREFILL", raising=False)
    monkeypatch.setattr(qsa_indexer_select, "qsa_indexer_select_nax_available", lambda: nax)
    monkeypatch.setattr(qwen4_exp, "qsa_prefill_lane_auto_supported", lambda: consumer)
    calls = []
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_mpp_compile_ok", lambda: calls.append(True) or mpp)
    assert qwen4_exp._qsa_prefill_enabled() is expected
    assert bool(calls) is (nax and consumer)
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "0")
    assert qwen4_exp._qsa_prefill_enabled() is False
