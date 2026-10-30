from __future__ import annotations

import json

from RetailPilot.agent import chat


if __name__ == "__main__":
    result = chat(
        "trace-demo",
        "帮我查订单 JD20260801，顺便推荐 50 美元以内的无线耳机",
        "demo-user",
    )
    print(json.dumps(result["decisions"], ensure_ascii=False, indent=2))
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    print(result["reply"])
