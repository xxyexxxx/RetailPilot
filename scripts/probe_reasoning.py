"""Explicit live capability probe; never changes deployment configuration."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from pydantic import BaseModel, ConfigDict

from RetailPilot.multimodal_model import (
    ImageInput,
    capability_probe,
    config,
    endpoint_fingerprint,
    gateway,
)


class TextProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    marker: Literal["retailpilot"]


class VisionProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    color: Literal["green"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = config("evidence_assessment")
    token = capability_probe.set(True)
    report = {
        "model": c.model,
        "endpoint_fingerprint": endpoint_fingerprint(c),
        "real_model": True,
        "text_verified": False,
        "vision_verified": False,
    }
    try:
        gateway.generate(
            task_type="evidence_assessment",
            messages=[
                {"role": "user", "content": 'Return JSON {"marker":"retailpilot"}'}
            ],
            output_schema=TextProbe,
        )
        report["text_verified"] = True
        image = Image.new("RGB", (64, 64), "#00ff00")
        data = io.BytesIO()
        image.save(data, "PNG")
        gateway.generate(
            task_type="evidence_assessment",
            messages=[
                {
                    "role": "user",
                    "content": "What is the single color of the attached image? Return JSON color.",
                }
            ],
            images=[ImageInput(source_kind="probe", data=data.getvalue())],
            output_schema=VisionProbe,
        )
        report["vision_verified"] = True
    except Exception as exc:  # noqa: BLE001 - safe probe status only
        report["error"] = type(exc).__name__
    finally:
        capability_probe.reset(token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["vision_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
