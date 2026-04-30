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
DEFAULT_SMART_THINKING_LEVEL = "high"
DEFAULT_FAST_THINKING_LEVEL = "high"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 500.0
DEFAULT_MAX_RETRIES = 3
SUPPORTED_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
DISABLED_THINKING_LEVEL_VALUES = frozenset({"", "default", "auto", "unset", "none", "off"})

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
    if "gemini-3.1-pro" in str(model_name or "").lower():
        return 1
    return temperature


def _configured_env_value(*names: str) -> str | None:
    for name in names:
        if name in os.environ:
            return os.environ.get(name)
    return None


def _resolve_timeout_setting(request_timeout: float | None) -> float:
    if request_timeout is not None:
        return float(request_timeout)
    env_value = _configured_env_value("PAPERALCHEMY_LLM_TIMEOUT_SECONDS")
    if env_value is not None and str(env_value).strip():
        return float(env_value)
    return DEFAULT_REQUEST_TIMEOUT_SECONDS


def _resolve_max_retries(retries: int | None) -> int:
    if retries is not None:
        return int(retries)
    env_value = _configured_env_value("PAPERALCHEMY_LLM_MAX_RETRIES")
    if env_value is not None and str(env_value).strip():
        return int(env_value)
    return DEFAULT_MAX_RETRIES


def _supports_thinking_level(model_name: str) -> bool:
    return "gemini-3" in model_name.lower()


def _allowed_thinking_levels_for_model(model_name: str) -> frozenset[str]:
    normalized = model_name.lower()
    if not _supports_thinking_level(model_name):
        return frozenset()
    if "pro-image" in normalized:
        return frozenset({"high"})
    if "flash-image" in normalized:
        return frozenset({"minimal", "high"})
    if "pro" in normalized:
        return frozenset({"low", "medium", "high"})
    if "flash" in normalized:
        return SUPPORTED_THINKING_LEVELS
    return SUPPORTED_THINKING_LEVELS


def _normalize_thinking_level(raw_value: str | None, *, source: str) -> str | None:
    if raw_value is None:
        return None

    value = str(raw_value).strip().lower()
    if value in DISABLED_THINKING_LEVEL_VALUES:
        return None
    if value not in SUPPORTED_THINKING_LEVELS:
        allowed = ", ".join(sorted(SUPPORTED_THINKING_LEVELS))
        raise ValueError(f"{source} must be one of {allowed}, or 'default' to use the provider default.")
    return value


def _resolve_thinking_level(
    *,
    model_name: str,
    use_smart_model: bool,
    thinking_level: str | None,
) -> str | None:
    env_source_name = (
        "PAPERALCHEMY_SMART_THINKING_LEVEL"
        if use_smart_model
        else "PAPERALCHEMY_FAST_THINKING_LEVEL"
    )
    env_value = _configured_env_value("PAPERALCHEMY_THINKING_LEVEL", env_source_name)

    if not _supports_thinking_level(model_name):
        configured_value = thinking_level if thinking_level is not None else env_value
        normalized = _normalize_thinking_level(configured_value, source="thinking_level")
        if normalized is not None:
            raise ValueError(
                "thinking_level is only supported for Gemini 3 and later models; "
                "use the provider default for older Gemini models."
            )
        return None

    default_level = DEFAULT_SMART_THINKING_LEVEL if use_smart_model else DEFAULT_FAST_THINKING_LEVEL
    raw_value = thinking_level if thinking_level is not None else env_value
    if raw_value is None:
        raw_value = default_level

    normalized = _normalize_thinking_level(raw_value, source="thinking_level")
    if normalized is None:
        return None

    allowed_levels = _allowed_thinking_levels_for_model(model_name)
    if normalized not in allowed_levels:
        allowed = ", ".join(sorted(allowed_levels))
        raise ValueError(
            f"thinking_level={normalized!r} is not supported by {model_name!r}; "
            f"allowed values: {allowed}."
        )
    return normalized


def get_llm(
    temperature: float = 0,
    use_smart_model: bool = True,
    *,
    request_timeout: float | None = None,
    retries: int | None = None,
    streaming: bool | None = None,
    thinking_level: str | None = None,
):
    vertex_runtime = _load_vertex_runtime()
    provider = "vertex" if vertex_runtime is not None else "api_key"
    model_name = _resolve_model_name(use_smart_model=use_smart_model, provider=provider)
    effective_temperature = _resolve_effective_temperature(
        temperature,
        provider=provider,
        model_name=model_name,
        use_smart_model=use_smart_model,
    )
    timeout_setting = _resolve_timeout_setting(request_timeout)
    max_retries = _resolve_max_retries(retries)
    resolved_thinking_level = _resolve_thinking_level(
        model_name=model_name,
        use_smart_model=use_smart_model,
        thinking_level=thinking_level,
    )

    print(
        f"[PaperAlchemy] Initializing Gemini via {provider}: "
        f"{model_name} (temp={effective_temperature}, "
        f"thinking_level={resolved_thinking_level or 'default'}, "
        f"timeout={timeout_setting:g}s, retries={max_retries})"
    )

    common_kwargs = {
        "model": model_name,
        "temperature": effective_temperature,
        "max_retries": max_retries,
        "timeout": timeout_setting,
        "streaming": True if streaming is None else bool(streaming),
        "safety_settings": {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        },
        "convert_system_message_to_human": True,
    }
    if resolved_thinking_level is not None:
        common_kwargs["thinking_level"] = resolved_thinking_level

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

