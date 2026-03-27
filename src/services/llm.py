import json
import os
import ssl
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Force-disable SSL cert validation for the existing local proxy/debug setup.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_SMART_MODEL = "gemini-3.1-pro-preview"
DEFAULT_VERTEX_FAST_MODEL = "gemini-3-flash-preview"
DEFAULT_API_KEY_SMART_MODEL = "gemini-3.1-pro-preview"
DEFAULT_API_KEY_FAST_MODEL = "gemini-3-flash-preview"

proxy_url = os.getenv("HTTPS_PROXY") or "http://127.0.0.1:7890"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
os.environ["all_proxy"] = proxy_url
os.environ["CURL_CA_BUNDLE"] = ""


def _is_service_account_json(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".json":
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        payload.get("type") == "service_account"
        and bool(payload.get("project_id"))
        and bool(payload.get("client_email"))
        and bool(payload.get("private_key"))
    )


def _discover_vertex_credentials_path() -> Path | None:
    explicit_candidates = [
        os.getenv("VERTEX_SERVICE_ACCOUNT_JSON"),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    ]
    for raw_path in explicit_candidates:
        clean = str(raw_path or "").strip().strip('"')
        if not clean:
            continue
        candidate = Path(clean)
        if _is_service_account_json(candidate):
            return candidate

    discovered_candidates = [
        path
        for path in sorted(PROJECT_ROOT.glob("*.json"))
        if _is_service_account_json(path)
    ]
    if len(discovered_candidates) == 1:
        return discovered_candidates[0]
    return None


def _load_vertex_runtime() -> dict[str, Any] | None:
    credentials_path = _discover_vertex_credentials_path()
    if credentials_path is None:
        return None

    try:
        raw_payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=VERTEX_SCOPES,
        )
    except Exception as exc:
        raise ValueError(f"Detected a Vertex service account JSON, but failed to load it: {exc}") from exc

    project_id = str(
        os.getenv("VERTEX_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or raw_payload.get("project_id")
        or ""
    ).strip()
    if not project_id:
        raise ValueError(
            "Vertex service account JSON is missing project_id, and no environment override was provided."
        )

    location = str(
        os.getenv("VERTEX_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or DEFAULT_VERTEX_LOCATION
    ).strip()

    return {
        "credentials_path": credentials_path,
        "credentials": credentials,
        "project_id": project_id,
        "location": location,
    }


def _resolve_model_name(use_smart_model: bool, *, provider: str) -> str:
    if use_smart_model:
        env_override = os.getenv("PAPERALCHEMY_SMART_MODEL")
        if env_override:
            return str(env_override).strip()
        return DEFAULT_VERTEX_SMART_MODEL if provider == "vertex" else DEFAULT_API_KEY_SMART_MODEL

    env_override = os.getenv("PAPERALCHEMY_FAST_MODEL")
    if env_override:
        return str(env_override).strip()
    return DEFAULT_VERTEX_FAST_MODEL if provider == "vertex" else DEFAULT_API_KEY_FAST_MODEL


def _resolve_effective_temperature(
    temperature: float,
    *,
    provider: str,
    model_name: str,
    use_smart_model: bool,
) -> float:
    if provider == "vertex" and use_smart_model and model_name == DEFAULT_VERTEX_SMART_MODEL:
        return 1
    return temperature


def get_llm(temperature: float = 0, use_smart_model: bool = True):
    vertex_runtime = _load_vertex_runtime()
    provider = "vertex" if vertex_runtime is not None else "api_key"
    model_name = _resolve_model_name(use_smart_model=use_smart_model, provider=provider)
    effective_temperature = _resolve_effective_temperature(
        temperature,
        provider=provider,
        model_name=model_name,
        use_smart_model=use_smart_model,
    )
    timeout_setting = 500.0

    print(
        f"[PaperAlchemy] Initializing Gemini via {provider}: "
        f"{model_name} (temp={effective_temperature})"
    )

    common_kwargs = {
        "model": model_name,
        "temperature": effective_temperature,
        "retries": 3,
        "request_timeout": timeout_setting,
        "streaming": True,
        "safety_settings": {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        },
        "convert_system_message_to_human": True,
    }

    if vertex_runtime is not None:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", str(vertex_runtime["project_id"]))
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", str(vertex_runtime["location"]))
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(vertex_runtime["credentials_path"]))
        return ChatGoogleGenerativeAI(
            **common_kwargs,
            credentials=vertex_runtime["credentials"],
            vertexai=True,
            project=str(vertex_runtime["project_id"]),
            location=str(vertex_runtime["location"]),
        )

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "No usable Gemini credentials were found: neither a Vertex service account JSON nor GOOGLE_API_KEY is available."
        )

    return ChatGoogleGenerativeAI(**common_kwargs)

