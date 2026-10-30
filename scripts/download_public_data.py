from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DATASETS: dict[str, dict[str, str]] = {
    "shopsimulator": {
        "url": (
            "https://raw.githubusercontent.com/ShopAgent-Team/ShopSimulator/main/"
            "shop_env/data/fine_items_eval_train_all.json.gz"
        ),
        "filename": "shopsimulator/fine_items_eval_train_all.json.gz",
        "upstream": "https://github.com/ShopAgent-Team/ShopSimulator",
        "description": "公开中文购物环境的商品和任务数据",
    },
    "jddc-uss": {
        "url": (
            "https://raw.githubusercontent.com/sunnweiwei/"
            "user-satisfaction-simulation/master/dataset/JDDC.txt"
        ),
        "filename": "jddc-uss/JDDC.txt",
        "upstream": "https://github.com/sunnweiwei/user-satisfaction-simulation",
        "description": "USS 发布的 JDDC 真实中文电商客服多轮对话子集",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    name: str, output_root: Path, *, force: bool = False
) -> dict[str, Any]:
    spec = DATASETS[name]
    target = output_root / spec["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return {
            "dataset": name,
            "path": str(target.resolve()),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
            "downloaded": False,
            "upstream": spec["upstream"],
        }

    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        spec["url"], headers={"User-Agent": "RetailPilot/1.0"}
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "dataset": name,
        "path": str(target.resolve()),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "downloaded": True,
        "upstream": spec["upstream"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public research datasets into ignored local cache"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[*DATASETS, "all"],
        default=[],
        help="Repeat to select datasets; default is all",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data" / "external"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--accept-upstream-terms",
        action="store_true",
        help="Required: acknowledge that upstream data terms must be reviewed",
    )
    args = parser.parse_args()
    if not args.accept_upstream_terms:
        raise SystemExit(
            "Refusing to download third-party data without --accept-upstream-terms. "
            "The upstream repositories do not expose a clear LICENSE file; use for "
            "local research only and review their current terms before redistribution."
        )
    selected = args.dataset or ["all"]
    names = list(DATASETS) if "all" in selected else list(dict.fromkeys(selected))
    output_root = args.output_root.resolve()
    records = [download_dataset(name, output_root, force=args.force) for name in names]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "redistribution": "raw files are local cache and are not committed",
        "license_note": (
            "No clear LICENSE file was found in the upstream repositories at integration "
            "time. Review upstream terms before use or redistribution."
        ),
        "datasets": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
