from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RetailPilot.config import settings
from RetailPilot.customer_eval import load_customer_cases, run_customer_cases
from RetailPilot.llm_config import judge_config_status


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _markdown(report: dict) -> str:
    summary = report["summary"]
    semantic = summary["semantic_sidecar"]
    lines = [
        "# Customer-service Agent Harness",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Harness: `{report['harness_version']}`",
        f"- Cases: {summary['cases_passed']}/{summary['cases']}",
        f"- Turns: {summary['turns_passed']}/{summary['turns']} ({summary['turn_pass_rate']:.1%})",
        f"- Failure counts: `{json.dumps(summary['failure_counts'], ensure_ascii=False)}`",
        f"- Real LLM Judge calls / evaluable / not-evaluable: "
        f"{semantic['llm_judged_turns']} / {semantic['llm_evaluable_turns']} / "
        f"{semantic['llm_not_evaluable_turns']}; pass "
        f"{semantic['turns_passed']} ({_rate(semantic['turn_pass_rate'])})",
        f"- Deterministic / fallback / not-run (excluded from semantic metrics): "
        f"{semantic['deterministic_turns']} / "
        f"{semantic['deterministic_fallback_turns']} / {semantic['not_run_turns']}",
        f"- Judge modes: `{json.dumps(semantic['judge_mode_counts'], ensure_ascii=False)}`",
        f"- Real LLM failures: `{json.dumps(semantic['failure_counts'], ensure_ascii=False)}`",
        "",
        "| Case | Pass | Failed turns |",
        "|---|---:|---|",
    ]
    for case in report["cases"]:
        failed = [
            f"T{turn['turn_index']}:{','.join(turn['failure_types'])}"
            for turn in case["turns"]
            if not turn["pass"]
        ]
        lines.append(f"| {case['case_id']} | {'yes' if case['pass'] else 'no'} | {'; '.join(failed) or '-'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trace-aware multi-turn customer-service cases")
    parser.add_argument(
        "--cases",
        type=Path,
        default=settings.root_dir / "config" / "customer-service-cases.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--llm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the independent per-turn LLM judge (default: on; --no-llm-judge disables)",
    )
    parser.add_argument(
        "--require-llm-judge",
        action="store_true",
        help="Exit non-zero unless every turn was judged by a real LLM",
    )
    args = parser.parse_args()
    if args.llm_judge:
        judge_status = judge_config_status()
        if not judge_status.get("configured"):
            raise SystemExit(
                "LLM judge is mandatory by default and no independent JUDGE_API_KEY "
                f"is configured: {judge_status.get('error') or 'unknown error'}. "
                "Set JUDGE_* explicitly or pass --no-llm-judge to opt out."
            )
    cases = load_customer_cases(args.cases)
    report = run_customer_cases(cases, use_llm_judge=args.llm_judge)
    output = args.output or (
        settings.artifacts_dir
        / "customer_service_runs"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, ensure_ascii=False))
    if args.require_llm_judge:
        if not args.llm_judge:
            raise SystemExit("--require-llm-judge also requires --llm-judge")
        modes = report["summary"]["semantic_sidecar"]["judge_mode_counts"]
        non_llm = sum(count for mode, count in modes.items() if mode != "llm")
        if non_llm or not modes.get("llm"):
            raise SystemExit(f"Real LLM Judge requirement failed: judge modes={modes}")


if __name__ == "__main__":
    main()
