from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, TypedDict


class FeedbackImagePayload(TypedDict):
    name: str
    path: str
    mime_type: str
    data_url: str


class HumanFeedbackPayload(TypedDict):
    text: str
    images: list[FeedbackImagePayload]


def empty_human_feedback() -> HumanFeedbackPayload:
    return {"text": "", "images": []}


def _coerce_uploaded_paths(value: Any) -> list[Path]:
    if value is None:
        return []

    if isinstance(value, (str, Path)):
        return [Path(value)]

    if isinstance(value, list):
        paths: list[Path] = []
        for item in value:
            paths.extend(_coerce_uploaded_paths(item))
        return paths

    candidate_path = getattr(value, "name", None)
    if isinstance(candidate_path, str) and candidate_path.strip():
        return [Path(candidate_path)]

    return []


def _encode_image_to_data_url(image_path: Path) -> tuple[str, str] | None:
    try:
        image_bytes = image_path.read_bytes()
    except Exception:
        return None

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return mime_type, f"data:{mime_type};base64,{encoded}"


def build_human_feedback_payload(
    text: str = "",
    image_paths: Any = None,
) -> HumanFeedbackPayload:
    normalized_text = str(text or "").strip()
    images: list[FeedbackImagePayload] = []

    seen_paths: set[str] = set()
    for raw_path in _coerce_uploaded_paths(image_paths):
        resolved_path = raw_path.expanduser().resolve()
        resolved_text = str(resolved_path)
        if resolved_text in seen_paths or not resolved_path.exists() or not resolved_path.is_file():
            continue

        encoded_image = _encode_image_to_data_url(resolved_path)
        if not encoded_image:
            continue

        mime_type, data_url = encoded_image
        images.append(
            {
                "name": resolved_path.name,
                "path": resolved_text,
                "mime_type": mime_type,
                "data_url": data_url,
            }
        )
        seen_paths.add(resolved_text)

    return {"text": normalized_text, "images": images}


def normalize_human_feedback(value: Any) -> HumanFeedbackPayload:
    if isinstance(value, str):
        return build_human_feedback_payload(text=value)

    if not isinstance(value, dict):
        return empty_human_feedback()

    text = str(value.get("text") or "").strip()
    raw_images = value.get("images")
    images: list[FeedbackImagePayload] = []
    if isinstance(raw_images, list):
        for item in raw_images:
            if isinstance(item, dict):
                image_path = str(item.get("path") or "").strip()
                data_url = str(item.get("data_url") or "").strip()
                mime_type = str(item.get("mime_type") or "").strip() or "image/png"
                image_name = str(item.get("name") or Path(image_path or "upload").name)
            else:
                image_path = str(item or "").strip()
                data_url = ""
                mime_type = "image/png"
                image_name = Path(image_path or "upload").name

            if not data_url and image_path:
                encoded_image = _encode_image_to_data_url(Path(image_path))
                if encoded_image:
                    mime_type, data_url = encoded_image

            if not data_url:
                continue

            images.append(
                {
                    "name": image_name,
                    "path": image_path,
                    "mime_type": mime_type,
                    "data_url": data_url,
                }
            )

    return {"text": text, "images": images}


def extract_human_feedback_text(value: Any) -> str:
    return normalize_human_feedback(value)["text"]


def extract_human_feedback_images(value: Any) -> list[FeedbackImagePayload]:
    return normalize_human_feedback(value)["images"]


def has_human_feedback(value: Any) -> bool:
    normalized = normalize_human_feedback(value)
    return bool(normalized["text"] or normalized["images"])


def build_multimodal_message_content(
    text: str,
    images: list[FeedbackImagePayload] | None = None,
) -> list[dict[str, str]]:
    message_content: list[dict[str, str]] = [{"type": "text", "text": str(text or "").strip()}]
    for image in images or []:
        data_url = str(image.get("data_url") or "").strip()
        if data_url:
            message_content.append({"type": "image_url", "image_url": data_url})
    return message_content

