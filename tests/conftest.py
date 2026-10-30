from __future__ import annotations

import sqlite3

import pytest

from RetailPilot.config import settings
from RetailPilot.database import init_db


@pytest.fixture(scope="session", autouse=True)
def isolated_database(tmp_path_factory):
    original_llm_mode = settings.llm_mode
    original_intent_mode = settings.intent_mode
    original_embedding_model = settings.embedding_model
    original_policy_vector_backend = settings.policy_vector_backend
    original_policy_reranker_enabled = settings.policy_reranker_enabled
    original_allow_open_api = settings.allow_open_api
    # Unit/integration tests are deterministic. Individual LLM tests install a
    # fake model and explicitly opt back into the LLM path.
    settings.llm_mode = "off"
    settings.intent_mode = "rules"
    settings.embedding_model = "hash"
    settings.policy_vector_backend = "disabled"
    settings.policy_reranker_enabled = False
    settings.allow_open_api = True
    source = settings.database_path
    target = tmp_path_factory.mktemp("db") / "retailpilot-test.db"
    source_connection = sqlite3.connect(source)
    target_connection = sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
        target_connection.execute("DELETE FROM messages")
        target_connection.execute("DELETE FROM sessions")
        target_connection.execute("DELETE FROM search_logs")
        target_connection.execute("DELETE FROM interactions")
        target_connection.commit()
    finally:
        target_connection.close()
        source_connection.close()
    settings.database_path = target
    init_db(target)
    yield target
    settings.llm_mode = original_llm_mode
    settings.intent_mode = original_intent_mode
    settings.embedding_model = original_embedding_model
    settings.policy_vector_backend = original_policy_vector_backend
    settings.policy_reranker_enabled = original_policy_reranker_enabled
    settings.allow_open_api = original_allow_open_api
