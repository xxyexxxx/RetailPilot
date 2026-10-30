"""
公共工具：LLM 输出容错解析、中文数字映射。
拆分自 agent.py：被 nlu.py（槽位解析）与其他意图/检索路径共享。
"""
import json
import re

_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _parse_json(resp: str) -> dict:
    """LLM 输出 → dict：容忍 markdown 代码块和多余文字，抽第一个花括号块。"""
    try:
        return json.loads(resp)
    except Exception:
        m = re.search(r"\{.*\}", resp, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}
