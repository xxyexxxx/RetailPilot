from __future__ import annotations

import argparse
import json

import uvicorn

from RetailPilot.agent import chat
from RetailPilot.config import settings
from RetailPilot.database import catalog_ready
from RetailPilot.evaluation import run_evaluation


DEMO_TURNS = [
    "推荐 50 美元以内评分高的无线耳机",
    "对比前两款",
    "第一款怎么样",
    "再便宜一点",
    "帮我查一下订单 JD20260801 到哪了",
]


def run_demo() -> None:
    session_id = "cli-demo"
    for message in DEMO_TURNS:
        result = chat(session_id, message, "demo-user")
        print(f"\n用户: {message}")
        print(f"助手: {result['reply']}")
        if result["products"]:
            print("商品:", ", ".join(product["sku"] for product in result["products"][:5]))


def run_chat() -> None:
    session_id = "cli-interactive"
    print("RetailPilot，输入 exit 退出。")
    while True:
        try:
            message = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message or message.lower() in {"exit", "quit", "退出"}:
            break
        result = chat(session_id, message, "demo-user")
        print(f"助手: {result['reply']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPilot")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["serve", "demo", "chat", "eval"],
        default="serve",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not catalog_ready():
        raise SystemExit(
            "商品库尚未初始化，请先运行: python scripts/bootstrap_data.py"
        )
    if args.command == "serve":
        uvicorn.run(
            "RetailPilot.api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    elif args.command == "demo":
        run_demo()
    elif args.command == "chat":
        run_chat()
    else:
        report = run_evaluation()
        summary = {
            mode: payload["metrics"] for mode, payload in report["modes"].items()
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"report: {settings.artifacts_dir / 'eval_report.md'}")


if __name__ == "__main__":
    main()
