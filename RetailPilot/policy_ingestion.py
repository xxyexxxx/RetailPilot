from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .database import (
    db_session,
    init_db,
    json_dumps,
    refresh_policy_fts,
    utc_now,
)


_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$\n?")
_TEXT_UNIT = re.compile(r"[^\n。！？；;]+[。！？；;]?|\n+")


@dataclass(slots=True)
class PolicyDocumentInput:
    doc_id: str
    policy_id: str
    topic: str
    title: str
    content: str
    source_name: str
    source_uri: str = ""
    platform: str = "demo-mall"
    jurisdiction: str = "*"
    language: str = "zh-CN"
    category_scope: str = "*"
    seller_scope: str = "*"
    product_type_scope: str = "*"
    audience: str = "customer"
    effective_from: str = "0001-01-01"
    effective_to: str = ""
    version: str = "1.0"
    status: str = "active"
    supersedes_doc_id: str = ""
    authority: str = "demo"
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def approximate_token_count(text: str) -> int:
    """Cheap language-neutral sizing used only for structural chunking."""

    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9._+#/-]*", text))
    return max(1, round(cjk / 1.7) + words)


def _digest(text: str, size: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]


def _document_title(markdown: str, fallback: str) -> str:
    for match in _HEADING.finditer(markdown):
        if len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback


def _sections(markdown: str, fallback_title: str) -> list[dict[str, Any]]:
    matches = list(_HEADING.finditer(markdown))
    if not matches:
        content = markdown.strip()
        return [
            {
                "path": fallback_title,
                "content": content,
                "start": markdown.find(content) if content else 0,
            }
        ]

    stack: list[str] = []
    result: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        stack = stack[: level - 1]
        stack.append(heading)
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        raw_body = markdown[body_start:body_end]
        body = raw_body.strip()
        if not body:
            continue
        leading = len(raw_body) - len(raw_body.lstrip())
        result.append(
            {
                "path": " > ".join(stack),
                "content": body,
                "start": body_start + leading,
            }
        )
    if result:
        return result
    content = _HEADING.sub("", markdown).strip()
    return [{"path": fallback_title, "content": content, "start": markdown.find(content)}]


def _units(text: str) -> list[str]:
    return [match.group(0).strip() for match in _TEXT_UNIT.finditer(text) if match.group(0).strip()]


def _child_windows(
    text: str,
    *,
    target_tokens: int = 280,
    max_tokens: int = 350,
    overlap_tokens: int = 60,
) -> list[str]:
    units = _units(text)
    if not units:
        return [text]
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = approximate_token_count(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            windows.append("".join(current).strip())
            overlap: list[str] = []
            overlap_count = 0
            for previous in reversed(current):
                overlap.insert(0, previous)
                overlap_count += approximate_token_count(previous)
                if overlap_count >= overlap_tokens:
                    break
            current = overlap
            current_tokens = overlap_count
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= target_tokens:
            windows.append("".join(current).strip())
            overlap = []
            overlap_count = 0
            for previous in reversed(current):
                overlap.insert(0, previous)
                overlap_count += approximate_token_count(previous)
                if overlap_count >= overlap_tokens:
                    break
            current = overlap
            current_tokens = overlap_count
    tail = "".join(current).strip()
    if tail and (not windows or tail != windows[-1]):
        windows.append(tail)
    return list(dict.fromkeys(item for item in windows if item))


def _parent_windows(
    text: str, *, target_tokens: int = 1000, max_tokens: int = 1200
) -> list[str]:
    units = _units(text)
    if not units:
        return [text]
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = approximate_token_count(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            windows.append("".join(current).strip())
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= target_tokens:
            windows.append("".join(current).strip())
            current = []
            current_tokens = 0
    tail = "".join(current).strip()
    if tail:
        windows.append(tail)
    return [item for item in windows if item]


def _contextualized(document: PolicyDocumentInput, section_path: str, text: str) -> str:
    scope = " | ".join(
        value
        for value in (
            document.topic,
            document.platform,
            document.jurisdiction,
            document.category_scope,
        )
        if value and value != "*"
    )
    return f"{document.title} | {section_path} | {scope} | {text}".strip(" |")


def chunk_policy_document(document: PolicyDocumentInput) -> list[dict[str, Any]]:
    """Build structural leaf chunks; only long sections receive parent/child rows."""

    document.title = _document_title(document.content, document.title)
    chunks: list[dict[str, Any]] = []
    ordinal = 0
    for section in _sections(document.content, document.title):
        text = str(section["content"])
        section_path = str(section["path"])
        base_start = int(section["start"])
        if approximate_token_count(text) <= 350:
            content_hash = _digest(text)
            chunks.append(
                {
                    "chunk_id": f"{document.doc_id}:leaf:{content_hash}",
                    "doc_id": document.doc_id,
                    "parent_chunk_id": None,
                    "chunk_type": "leaf",
                    "ordinal": ordinal,
                    "section_path": section_path,
                    "page_no": None,
                    "char_start": base_start,
                    "char_end": base_start + len(text),
                    "content": text,
                    "contextualized_content": _contextualized(
                        document, section_path, text
                    ),
                    "token_count": approximate_token_count(text),
                    "content_hash": content_hash,
                    "metadata_json": "{}",
                }
            )
            ordinal += 1
            continue

        # A subsection is a 600-1200 token parent when possible. Oversized
        # sections become several bounded parents. Only their children are
        # indexed; retrieval expands a matched child back to its parent.
        section_search_from = 0
        for parent_index, parent_text in enumerate(_parent_windows(text)):
            parent_relative_start = text.find(parent_text, section_search_from)
            if parent_relative_start < 0:
                parent_relative_start = text.find(parent_text)
            parent_relative_start = max(0, parent_relative_start)
            section_search_from = parent_relative_start + len(parent_text)
            parent_path = (
                section_path
                if parent_index == 0
                else f"{section_path} [{parent_index + 1}]"
            )
            parent_hash = _digest(f"{parent_path}\n{parent_text}")
            parent_id = f"{document.doc_id}:parent:{parent_hash}"
            chunks.append(
                {
                    "chunk_id": parent_id,
                    "doc_id": document.doc_id,
                    "parent_chunk_id": None,
                    "chunk_type": "parent",
                    "ordinal": ordinal,
                    "section_path": parent_path,
                    "page_no": None,
                    "char_start": base_start + parent_relative_start,
                    "char_end": base_start + parent_relative_start + len(parent_text),
                    "content": parent_text,
                    "contextualized_content": _contextualized(
                        document, parent_path, parent_text
                    ),
                    "token_count": approximate_token_count(parent_text),
                    "content_hash": parent_hash,
                    "metadata_json": "{}",
                }
            )
            ordinal += 1
            child_search_from = 0
            for child in _child_windows(parent_text):
                child_relative_start = parent_text.find(child, child_search_from)
                if child_relative_start < 0:
                    child_relative_start = parent_text.find(child)
                child_relative_start = max(0, child_relative_start)
                child_search_from = child_relative_start + max(1, len(child) // 2)
                child_hash = _digest(f"{parent_id}\n{child}")
                chunks.append(
                    {
                        "chunk_id": f"{document.doc_id}:child:{child_hash}",
                        "doc_id": document.doc_id,
                        "parent_chunk_id": parent_id,
                        "chunk_type": "child",
                        "ordinal": ordinal,
                        "section_path": parent_path,
                        "page_no": None,
                        "char_start": base_start
                        + parent_relative_start
                        + child_relative_start,
                        "char_end": base_start
                        + parent_relative_start
                        + child_relative_start
                        + len(child),
                        "content": child,
                        "contextualized_content": _contextualized(
                            document, parent_path, child
                        ),
                        "token_count": approximate_token_count(child),
                        "content_hash": child_hash,
                        "metadata_json": "{}",
                    }
                )
                ordinal += 1
    return chunks


def ingest_policy_documents(
    documents: Iterable[PolicyDocumentInput],
    *,
    database_path: Path | str | None = None,
    source_version: str = "",
) -> dict[str, Any]:
    materialized = list(documents)
    init_db(database_path)
    run_id = f"policy-{uuid.uuid4().hex}"
    chunk_count = 0
    doc_ids: list[str] = []
    with db_session(database_path) as connection:
        for document in materialized:
            checksum = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            chunks = chunk_policy_document(document)
            doc_ids.append(document.doc_id)
            connection.execute(
                """
                INSERT INTO policy_documents(
                    doc_id, policy_id, topic, title, content, source_name,
                    source_uri, platform, jurisdiction, language, category_scope,
                    seller_scope, product_type_scope, audience, effective_from,
                    effective_to, version, status, supersedes_doc_id, authority,
                    priority, checksum, metadata_json, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    policy_id=excluded.policy_id,
                    topic=excluded.topic,
                    title=excluded.title,
                    content=excluded.content,
                    source_name=excluded.source_name,
                    source_uri=excluded.source_uri,
                    platform=excluded.platform,
                    jurisdiction=excluded.jurisdiction,
                    language=excluded.language,
                    category_scope=excluded.category_scope,
                    seller_scope=excluded.seller_scope,
                    product_type_scope=excluded.product_type_scope,
                    audience=excluded.audience,
                    effective_from=excluded.effective_from,
                    effective_to=excluded.effective_to,
                    version=excluded.version,
                    status=excluded.status,
                    supersedes_doc_id=excluded.supersedes_doc_id,
                    authority=excluded.authority,
                    priority=excluded.priority,
                    checksum=excluded.checksum,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    document.doc_id,
                    document.policy_id,
                    document.topic,
                    document.title,
                    document.content,
                    document.source_name,
                    document.source_uri,
                    document.platform,
                    document.jurisdiction,
                    document.language,
                    document.category_scope,
                    document.seller_scope,
                    document.product_type_scope,
                    document.audience,
                    document.effective_from,
                    document.effective_to,
                    document.version,
                    document.status,
                    document.supersedes_doc_id,
                    document.authority,
                    document.priority,
                    checksum,
                    json_dumps(document.metadata),
                    utc_now(),
                ),
            )
            connection.execute(
                "DELETE FROM policy_chunks WHERE doc_id = ?", (document.doc_id,)
            )
            connection.executemany(
                """
                INSERT INTO policy_chunks(
                    chunk_id, doc_id, parent_chunk_id, chunk_type, ordinal,
                    section_path, page_no, char_start, char_end, content,
                    contextualized_content, token_count, content_hash,
                    metadata_json
                ) VALUES(
                    :chunk_id, :doc_id, :parent_chunk_id, :chunk_type, :ordinal,
                    :section_path, :page_no, :char_start, :char_end, :content,
                    :contextualized_content, :token_count, :content_hash,
                    :metadata_json
                )
                """,
                chunks,
            )
            chunk_count += len(chunks)
        refresh_policy_fts(connection, doc_ids)
        connection.execute(
            """
            INSERT INTO policy_ingestion_runs(
                run_id, source_name, source_version, document_count,
                chunk_count, manifest_json
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                ",".join(sorted({item.source_name for item in materialized})),
                source_version,
                len(materialized),
                chunk_count,
                json_dumps({"doc_ids": doc_ids}),
            ),
        )
    return {
        "run_id": run_id,
        "documents": len(materialized),
        "chunks": chunk_count,
        "source_version": source_version,
    }


def document_asdict(document: PolicyDocumentInput) -> dict[str, Any]:
    return asdict(document)
