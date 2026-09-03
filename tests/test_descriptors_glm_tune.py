"""Tune is enabled for the GLM MoE family.

`mtplx tune` drives `run_mtp_depth_sweep`, the same runner behind
`mtplx mtp-depth-sweep`.  That runner was measured end to end against a
forged GLM-4.7-Flash pack (glm4_moe_lite) on an M3 Max: AR plus depths 1-3,
reproducible across two runs, paired AR baselines within 0.15%.  The family
gate was a conservative allowlist rather than a technical limit, and it is
what made `mtplx forge build` exit 1 on GLM after a successful convert and
calibrate, leaving the artifact unbranded.

Depth is capped at D3 because GLM inherits
NATIVE_CONTRACT_DESCRIPTOR.draft_semantics (minimum 1, maximum 3).  Offering
D4+ would advertise depths the backend refuses -- the failure the qwen3_8
branch already guards against.
"""

from __future__ import annotations

from mtplx.backends.descriptors import (
    model_family_from_inspection,
    tune_policy_for_model,
)

GLM_LITE_INSPECTION = {
    "model_type": "glm4_moe_lite",
    "architecture": "Glm4MoeLiteForCausalLM",
    "num_hidden_layers": 47,
    "mtp_num_hidden_layers": 1,
}

GLM_MOE_INSPECTION = {
    "model_type": "glm4_moe",
    "architecture": "Glm4MoeForCausalLM",
    "num_hidden_layers": 46,
    "mtp_num_hidden_layers": 1,
}


def test_glm_lite_resolves_to_the_glm_family():
    assert model_family_from_inspection(GLM_LITE_INSPECTION) == "glm"


def test_tune_is_supported_for_glm_moe_lite():
    policy = tune_policy_for_model(inspection=GLM_LITE_INSPECTION)
    assert policy.supported is True
    assert policy.unsupported_reason is None


def test_tune_is_supported_for_plain_glm_moe():
    # Family-level, matching how kv_quant_policy_for_model treats GLM and how
    # every Qwen branch is written.
    policy = tune_policy_for_model(inspection=GLM_MOE_INSPECTION)
    assert policy.supported is True


def test_glm_tune_candidates_stop_at_depth_three():
    policy = tune_policy_for_model(inspection=GLM_LITE_INSPECTION)
    assert policy.candidates == ("AR", "D1", "D2", "D3")
    assert "D4" not in policy.candidates


def test_glm_tune_control_field_is_depth():
    policy = tune_policy_for_model(inspection=GLM_LITE_INSPECTION)
    assert policy.control_field == "depth"


def test_glm_policy_serialises_with_candidates_visible():
    # to_dict() blanks candidates when unsupported; forge's verify phase reads
    # this shape, so an enabled policy must expose the real candidate list.
    payload = tune_policy_for_model(inspection=GLM_LITE_INSPECTION).to_dict()
    assert payload["supported"] is True
    assert payload["candidates"] == ["AR", "D1", "D2", "D3"]
    assert payload["unsupported_reason"] is None


def test_unrelated_family_is_not_captured_by_the_glm_branch():
    # DeepSeek is the guard because it genuinely resolves to its own family
    # and is genuinely refused today.  Note an unrecognised model_type is NOT
    # a useful guard here: tune_policy_for_model resolves a descriptor first
    # and passes it to model_family_from_inspection, so unknown models fall
    # back to a Qwen descriptor and already report supported=True. That
    # pre-existing behaviour is out of scope for this change.
    policy = tune_policy_for_model(
        inspection={
            "model_type": "deepseek_v3",
            "architecture": "DeepseekV3ForCausalLM",
        }
    )
    assert policy.supported is False
    assert policy.unsupported_reason
