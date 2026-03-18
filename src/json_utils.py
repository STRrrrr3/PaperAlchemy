import json
from typing import Any


def to_pretty_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.dumps(value, indent=2, ensure_ascii=False)
