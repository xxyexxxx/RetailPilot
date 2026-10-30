from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from RetailPilot import config


@pytest.mark.parametrize("prefix", ["RETAILPILOT_", "ECOMMERCE_"])
def test_runtime_settings_accept_current_and_legacy_prefixes(prefix):
    values = {
        "DB_PATH": "data/custom.db",
        "CHECKPOINT_PATH": "data/custom-checkpoints.db",
        "EVAL_HOLDOUT_PATH": "data/custom-holdout.db",
        "ARTIFACTS_DIR": "custom-artifacts",
        "WEB_DIR": "custom-web",
        "VECTOR_DIM": "64",
        "EMBEDDING_MODEL": "test-model",
        "EMBEDDING_INDEX_DIR": "data/custom-index",
        "EMBEDDING_DEVICE": "cpu",
        "EMBEDDING_MIN_SCORE": "0.6",
        "PLATFORM": "test-mall",
        "CANDIDATE_POOL_LIMIT": "17",
        "EVAL_SAMPLE_RATE": "0.2",
        "API_KEY": "test-key",
        "ALLOW_OPEN_API": "true",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("RETAILPILOT_", "ECOMMERCE_"))
    }
    env.update({prefix + key: value for key, value in values.items()})
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; from dataclasses import asdict; "
                "from RetailPilot.config import settings; "
                "print(json.dumps(asdict(settings), default=str))"
            ),
        ],
        cwd=config.ROOT_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(result.stdout)
    for field, relative in {
        "database_path": "data/custom.db",
        "checkpoint_path": "data/custom-checkpoints.db",
        "eval_holdout_path": "data/custom-holdout.db",
        "artifacts_dir": "custom-artifacts",
        "web_dir": "custom-web",
        "embedding_index_dir": "data/custom-index",
    }.items():
        assert actual[field] == str(config.ROOT_DIR / relative)
    assert {
        field: actual[field]
        for field in (
            "vector_dim",
            "embedding_model",
            "embedding_device",
            "embedding_min_score",
            "commerce_platform",
            "candidate_pool_limit",
            "eval_sample_rate",
            "api_key",
            "allow_open_api",
        )
    } == {
        "vector_dim": 64,
        "embedding_model": "test-model",
        "embedding_device": "cpu",
        "embedding_min_score": 0.6,
        "commerce_platform": "test-mall",
        "candidate_pool_limit": 17,
        "eval_sample_rate": 0.2,
        "api_key": "test-key",
        "allow_open_api": True,
    }


@pytest.mark.parametrize("current", ["current-key", ""])
def test_current_configuration_overrides_legacy_even_when_empty(monkeypatch, current):
    monkeypatch.setenv("ECOMMERCE_API_KEY", "legacy-key")
    monkeypatch.setenv("RETAILPILOT_API_KEY", current)
    assert config._env("RETAILPILOT_API_KEY") == current


def test_existing_database_is_reused_until_a_new_database_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    current = data / "retailpilot.db"
    legacy = data / "ecommerce.db"
    assert config._default_database_path() == current
    legacy.touch()
    assert config._default_database_path() == legacy
    current.touch()
    assert config._default_database_path() == current
