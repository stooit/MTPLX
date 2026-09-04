"""The tune candidate runner reads the pack config to pick the verify lane.

_depth_sweep_native60 hardcoded verify_strategy="capture_commit" and the
linear-gdn-from-conv-tape capture backend for every candidate. Those are
qwen3-next structure lanes: their capture stack introspects the qwen3-next
DecoderLayer layout (input_layernorm et al.) and raises AttributeError on
Flash-Next hyper-connection layers. The server already coerces qwen4_exp to
'batched' at boot; the candidate runner now applies the same predicate.
"""

from __future__ import annotations

import json

from mtplx.commands.public import _model_config_is_qwen4_exp


def _write(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_flash_next_pack_is_recognised_from_its_text_config(tmp_path):
    path = _write(tmp_path, {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}})
    assert _model_config_is_qwen4_exp(path) is True


def test_top_level_model_type_alone_is_enough(tmp_path):
    path = _write(tmp_path, {"model_type": "qwen4_exp_text"})
    assert _model_config_is_qwen4_exp(path) is True


def test_dense_qwen3_5_pack_keeps_the_capture_commit_lane(tmp_path):
    path = _write(tmp_path, {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}})
    assert _model_config_is_qwen4_exp(path) is False


def test_missing_or_unreadable_config_is_not_flash_next(tmp_path):
    assert _model_config_is_qwen4_exp(str(tmp_path)) is False
    (tmp_path / "config.json").write_text("{not json")
    assert _model_config_is_qwen4_exp(str(tmp_path)) is False
