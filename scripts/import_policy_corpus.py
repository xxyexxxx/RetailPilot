from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.config import settings
from RetailPilot.policy_ingestion import (
    PolicyDocumentInput,
    ingest_policy_documents,
)


DEFAULT_MANIFEST = ROOT / "config" / "policy-corpus-sources.json"


def _title(markdown: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    return match.group(1).strip() if match else fallback


def _source_version(repository: str, timeout: float) -> str:
    response = requests.get(
        f"https://api.github.com/repos/{repository}/commits/main",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "RetailPilot/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["sha"])


def load_documents(
    manifest: dict[str, Any], *, platform: str, timeout: float
) -> tuple[list[PolicyDocumentInput], str]:
    repository = str(manifest["repository"])
    version = _source_version(repository, timeout)
    base_url = str(manifest["base_url"])
    documents: list[PolicyDocumentInput] = []
    for index, item in enumerate(manifest["documents"], start=1):
        path = str(item["path"])
        source_uri = f"{base_url}{path}"
        response = requests.get(
            source_uri,
            headers={"User-Agent": "RetailPilot/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        content = response.text
        slug = Path(path).stem.replace("_", "-")
        documents.append(
            PolicyDocumentInput(
                doc_id=f"csa-{index:02d}-{slug}",
                policy_id=f"csa-{slug}",
                topic=str(item["topic"]),
                title=_title(content, slug),
                content=content,
                source_name=repository,
                source_uri=source_uri,
                platform=platform,
                jurisdiction=str(item.get("jurisdiction") or "*"),
                category_scope=str(item.get("category_scope") or "*"),
                audience=str(item.get("audience") or "customer"),
                version=version[:12],
                authority=str(manifest.get("authority") or "community-demo"),
                priority=5,
                metadata={
                    "upstream_path": path,
                    "upstream_commit": version,
                    "demo_only": True,
                    "license_note": manifest.get("license_note", ""),
                },
            )
        )
    return documents, version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the attributed CustomServiceAgent policy corpus."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--database", type=Path, default=settings.database_path)
    parser.add_argument("--platform", default="")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    platform = args.platform or str(manifest["platform"])
    documents, version = load_documents(
        manifest, platform=platform, timeout=args.timeout
    )
    result = ingest_policy_documents(
        documents,
        database_path=args.database,
        source_version=version,
    )
    print(json.dumps({**result, "platform": platform}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
