"""Compare isolated old/new retrieval snapshots without writing user behavior to production."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = """
import json,sys,time
sys.path.insert(0,sys.argv[1])
from RetailPilot.database import init_db
from RetailPilot.retrieval import search_engine
from RetailPilot.policy import retrieve_policies
init_db()
out=[]
for item in json.load(sys.stdin):
    start=time.perf_counter()
    if item['domain']=='product':
        from RetailPilot.schemas import ParsedQuery
        query=ParsedQuery(raw_query=item['query'],rewritten_query=item['query'],
                          category=item.get('category'),grounded_target_terms=item.get('target_terms',[]))
        result=search_engine.search(query,limit=5)
        citations=result.get('citations',[])
        out.append({'skus':[p['sku'] for p in result['products']],
                    'routes':result.get('retrieval',{}).get('routes',{}),
                    'evidence':[e.get('unit_id',e.get('chunk_id')) for p in result['products']
                                for e in p.get('knowledge_evidence',p.get('asset_evidence',[]))],
                    'citations':[{k:c.get(k) for k in ('sku','doc_id','chunk_id','asset_id','field','value')} for c in citations],
                    'latency_ms':(time.perf_counter()-start)*1000})
    else:
        result=retrieve_policies(item['query'],topic=item['topic'])
        out.append({'documents':[d['doc_id'] for d in result['documents']],
                    'evidence':[d['chunk_id'] for d in result['documents']],
                    'citations':[d['citation'] for d in result['documents']],
                    'latency_ms':(time.perf_counter()-start)*1000})
print(json.dumps(out,ensure_ascii=False))
"""


def copy_db(source, target):
    with sqlite3.connect(source) as a, sqlite3.connect(target) as b:
        a.backup(b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-db", type=Path, required=True)
    parser.add_argument("--current-db", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="JSON list of domain/query and policy topic",
    )
    parser.add_argument("--baseline-ref", default="a18faa1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.queries.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as temporary:
        tmp = Path(temporary)
        archive = tmp / "legacy.tar"
        subprocess.run(
            [
                "git",
                "archive",
                args.baseline_ref,
                "--format=tar",
                "--output",
                str(archive),
            ],
            cwd=ROOT,
            check=True,
        )
        import tarfile

        old = tmp / "old"
        old.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(old, filter="data")
        results = []
        for code, source, name in (
            (old, args.baseline_db, "baseline"),
            (ROOT, args.current_db, "unified"),
        ):
            snapshot = tmp / (name + ".db")
            copy_db(source.resolve(), snapshot)
            env = {
                **os.environ,
                "RETAILPILOT_DB_PATH": str(snapshot),
                "RETAILPILOT_EMBEDDING_MODEL": "hash",
                "RETAILPILOT_KNOWLEDGE_PRODUCT_MODEL": "hash",
                "RETAILPILOT_KNOWLEDGE_POLICY_MODEL": "disabled",
                "POLICY_VECTOR_BACKEND": "disabled",
                "POLICY_RERANKER_ENABLED": "0",
                "RETAILPILOT_MM_STORE": str(ROOT / "data/assets"),
                "PYTHONUTF8": "1",
                "LLM_MODE": "off",
            }
            completed = subprocess.run(
                [sys.executable, "-c", DRIVER, str(code)],
                input=json.dumps(cases),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                cwd=code,
                timeout=120,
                check=True,
            )
            results.append(json.loads(completed.stdout))
    report = {
        "baseline_ref": args.baseline_ref,
        "isolated_snapshots": True,
        "real_models": False,
        "queries": [],
    }
    for case, before, after in zip(cases, *results, strict=True):
        report["queries"].append(
            {
                "query": case,
                "baseline": before,
                "unified": after,
                "same_skus": before.get("skus") == after.get("skus"),
                "same_evidence": before["evidence"] == after["evidence"],
                "same_citations": before["citations"] == after["citations"],
            }
        )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
