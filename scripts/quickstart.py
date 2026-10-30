from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-platform clone-to-running bootstrap for RetailPilot"
    )
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument(
        "--data",
        choices=["ci", "public"],
        default="ci",
        help="ci is fast/offline after dependencies; public adds ShopSimulator",
    )
    parser.add_argument(
        "--with-real-dialogues",
        action="store_true",
        help="Download the JDDC-derived 3,300-dialogue replay corpus",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument(
        "--database-path",
        type=Path,
        help="Override the configured database (existing local databases are reused)",
    )
    parser.add_argument("--accept-upstream-terms", action="store_true")
    args = parser.parse_args()

    venv = args.venv.resolve()
    python = _venv_python(venv)
    if not python.exists():
        _run([sys.executable, "-m", "venv", str(venv)], os.environ.copy())
    if not args.skip_install:
        _run(
            [str(python), "-m", "pip", "install", "-r", "requirements.txt"],
            os.environ.copy(),
        )

    env_file = ROOT / ".env"
    if not env_file.exists():
        shutil.copyfile(ROOT / ".env.example", env_file)

    env = os.environ.copy()
    if args.database_path is not None:
        database = args.database_path.resolve()
    else:
        configured_database = subprocess.check_output(
            [
                str(python),
                "-X",
                "utf8",
                "-c",
                "from RetailPilot.config import settings; print(settings.database_path)",
            ],
            cwd=ROOT,
            env=env,
            encoding="utf-8",
        ).strip()
        database = Path(configured_database)
    env.update(
        {
            "RETAILPILOT_DB_PATH": str(database),
            "RETAILPILOT_EMBEDDING_MODEL": "hash",
            "LLM_MODE": env.get("LLM_MODE", "off"),
            "INTENT_MODE": env.get("INTENT_MODE", "rules"),
        }
    )
    if database.exists() and not args.force_data:
        print(f"Existing database kept: {database}")
    else:
        _run([str(python), "scripts/bootstrap_ci.py"], env)

    selected: list[str] = []
    if args.data == "public":
        selected.append("shopsimulator")
    if args.with_real_dialogues:
        selected.append("jddc-uss")
    if selected:
        if not args.accept_upstream_terms:
            raise SystemExit(
                "Public datasets require --accept-upstream-terms after reviewing upstream terms."
            )
        command = [str(python), "scripts/download_public_data.py"]
        for name in selected:
            command.extend(["--dataset", name])
        command.append("--accept-upstream-terms")
        _run(command, env)
    if args.data == "public":
        _run(
            [
                str(python),
                "scripts/import_shopsimulator.py",
                "data/external/shopsimulator/fine_items_eval_train_all.json.gz",
                "--replace",
            ],
            env,
        )

    probe = (
        "from RetailPilot.database import init_db; "
        "from RetailPilot.catalog import catalog_stats; "
        "init_db(); print(catalog_stats())"
    )
    _run([str(python), "-c", probe], env)
    print("\nReady.")
    print(
        "PowerShell: "
        f"$env:RETAILPILOT_DB_PATH='{database}'; "
        "$env:RETAILPILOT_ALLOW_OPEN_API='true'; "
        f"{python} main.py serve"
    )
    print(
        "POSIX: "
        f"RETAILPILOT_DB_PATH='{database}' RETAILPILOT_ALLOW_OPEN_API=true "
        f"{python} main.py serve"
    )
    if args.with_real_dialogues:
        print(
            f"Replay: {python} scripts/run_multiturn_replay.py "
            "--input data/external/jddc-uss/JDDC.txt --format jddc-uss --limit 100"
        )


if __name__ == "__main__":
    main()
