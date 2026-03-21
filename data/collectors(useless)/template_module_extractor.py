#!/usr/bin/env python3
"""
Download and modularize top-ranked GitHub template repositories.

Pipeline:
1. Read ranked CSV produced by github_template_collector.py (or load from lock file)
2. Download repository source zip for top candidates
3. Extract component/style/token files into local module packs
4. Generate per-repo manifest, global module_index.json and reproducible lock file
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_BASE = "https://api.github.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPONENT_EXTS = {".tsx", ".jsx", ".vue", ".svelte", ".astro", ".html"}
STYLE_EXTS = {".css", ".scss", ".sass", ".less", ".styl", ".pcss"}
TOKEN_FILE_HINTS = {
    "tailwind.config.js",
    "tailwind.config.ts",
    "theme.ts",
    "theme.js",
    "tokens.ts",
    "tokens.js",
    "design-tokens.json",
}
SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    "storybook-static",
    "vendor",
    "tmp",
    "temp",
}
SKIP_FILE_PATTERNS = [
    ".test.",
    ".spec.",
    ".stories.",
    "__snapshots__",
    ".min.",
]


@dataclass
class CandidateRepo:
    full_name: str
    html_url: str
    final_score: float
    license_kind: str
    license_spdx: str
    default_branch: str = ""
    preferred_ref: str = ""


class GitHubFetcher:
    def __init__(
        self,
        token: str | None,
        timeout: float,
        retries: int,
        backoff: float,
        max_retry_delay: float = 30.0,
    ) -> None:
        self.token = token or ""
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff = max(0.1, float(backoff))
        self.max_retry_delay = max(1.0, float(max_retry_delay))

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "PaperAlchemy-Template-Module-Extractor",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _retry_delay(self, attempt: int, retry_after: str | None = None, reset_at: str | None = None) -> float:
        if retry_after:
            try:
                parsed = float(retry_after)
                if parsed > 0:
                    return min(parsed, self.max_retry_delay)
            except Exception:
                pass

        if reset_at:
            try:
                reset_epoch = int(reset_at)
                now_epoch = int(time.time())
                diff = reset_epoch - now_epoch + 1
                if diff > 0:
                    return min(float(diff), self.max_retry_delay)
            except Exception:
                pass

        delay = self.backoff * (2 ** max(attempt - 1, 0))
        if delay < 0.5:
            delay = 0.5
        return min(delay, self.max_retry_delay)

    def request_bytes(self, url: str, accept: str = "application/vnd.github+json") -> tuple[bytes, dict[str, str]]:
        last_error: Exception | None = None
        headers = self._headers(accept=accept)

        for attempt in range(1, self.retries + 1):
            req = Request(url=url, headers=headers, method="GET")
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    payload = resp.read()
                    meta = {k.lower(): v for k, v in resp.headers.items()}
                    return payload, meta
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                meta = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
                status = int(exc.code)
                rate_limited = status == 403 and str(meta.get("x-ratelimit-remaining", "")) == "0"
                transient = status in (429, 500, 502, 503, 504) or rate_limited

                if transient and attempt < self.retries:
                    delay = self._retry_delay(
                        attempt,
                        retry_after=meta.get("retry-after"),
                        reset_at=meta.get("x-ratelimit-reset"),
                    )
                    print(
                        f"[modules] warning: HTTP {status} for {url}. "
                        f"retry {attempt}/{self.retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                message = f"HTTP {status} for {url}"
                if body:
                    message += f" | {body[:300]}"
                raise RuntimeError(message) from exc
            except (URLError, ConnectionResetError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    delay = self._retry_delay(attempt)
                    print(
                        f"[modules] warning: network error for {url}: {exc}. "
                        f"retry {attempt}/{self.retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Network error for {url}: {exc}") from exc

        raise RuntimeError(f"Request failed for {url}: {last_error}")

    def request_json(self, url: str) -> tuple[dict[str, Any], dict[str, str]]:
        raw, headers = self.request_bytes(url=url, accept="application/vnd.github+json")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON response for {url}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected JSON shape for {url}")
        return payload, headers

    def get_default_branch(self, full_name: str) -> str:
        url = f"{API_BASE}/repos/{full_name}"
        payload, _ = self.request_json(url)
        branch = str(payload.get("default_branch") or "").strip()
        return branch or "main"

    def get_branch_head_sha(self, full_name: str, branch: str) -> str:
        safe_branch = quote(branch, safe="")
        url = f"{API_BASE}/repos/{full_name}/commits/{safe_branch}"
        payload, _ = self.request_json(url)
        sha = str(payload.get("sha") or "").strip()
        if not sha:
            raise RuntimeError(f"Cannot resolve head sha for {full_name}:{branch}")
        return sha

    def download_repo_zip_by_ref(self, full_name: str, ref: str) -> bytes:
        safe_ref = quote(ref, safe="")
        url = f"{API_BASE}/repos/{full_name}/zipball/{safe_ref}"
        raw, _ = self.request_bytes(url=url, accept="application/octet-stream")
        return raw


def load_env_file(path: Path) -> int:
    if not path.exists():
        return 0

    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value
            loaded += 1

    return loaded


def find_latest_csv(output_dir: Path) -> Path:
    files = sorted(output_dir.glob("templates_ranked_*.csv"))
    if not files:
        raise FileNotFoundError(f"No templates_ranked_*.csv found in {output_dir}")
    return files[-1]


def read_candidates_from_csv(
    csv_path: Path,
    top_n: int,
    min_score: float,
    allow_restricted_license: bool,
) -> list[CandidateRepo]:
    rows: list[CandidateRepo] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_name = str(row.get("full_name") or "").strip()
            if not full_name:
                continue

            try:
                score = float(row.get("final_score") or 0.0)
            except Exception:
                score = 0.0
            if score < min_score:
                continue

            license_kind = str(row.get("license_kind") or "").strip()
            if not allow_restricted_license and license_kind != "permissive":
                continue

            rows.append(
                CandidateRepo(
                    full_name=full_name,
                    html_url=str(row.get("html_url") or "").strip(),
                    final_score=score,
                    license_kind=license_kind,
                    license_spdx=str(row.get("license_spdx") or "").strip(),
                )
            )

    rows.sort(key=lambda x: x.final_score, reverse=True)
    if top_n > 0:
        rows = rows[:top_n]
    return rows


def read_candidates_from_lock(lock_path: Path, top_n: int) -> list[CandidateRepo]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid lock format: {lock_path}")

    rows: list[CandidateRepo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        full_name = str(entry.get("full_name") or "").strip()
        if not full_name:
            continue

        try:
            score = float(entry.get("final_score") or 0.0)
        except Exception:
            score = 0.0

        rows.append(
            CandidateRepo(
                full_name=full_name,
                html_url=str(entry.get("html_url") or "").strip(),
                final_score=score,
                license_kind=str(entry.get("license_kind") or "").strip(),
                license_spdx=str(entry.get("license_spdx") or "").strip(),
                default_branch=str(entry.get("default_branch") or "").strip(),
                preferred_ref=str(entry.get("resolved_ref") or "").strip(),
            )
        )

    rows.sort(key=lambda x: x.final_score, reverse=True)
    if top_n > 0:
        rows = rows[:top_n]
    return rows


def repo_slug(full_name: str) -> str:
    return full_name.replace("/", "__")


def extract_zip_to_dir(zip_data: bytes, dest_dir: Path) -> Path:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        zf.extractall(dest_dir)

    subdirs = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return dest_dir


def should_skip_file(path: Path) -> bool:
    lower = path.as_posix().lower()
    for token in SKIP_FILE_PATTERNS:
        if token in lower:
            return True
    return False


def text_or_empty(path: Path, max_bytes: int) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        raw = path.read_bytes()
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def infer_slot_candidates(file_path: Path, content: str) -> list[str]:
    haystack = (file_path.as_posix() + "\n" + content[:10000]).lower()
    slots: list[str] = []

    rules = {
        "hero": ["hero", "banner", "masthead"],
        "navigation": ["navbar", "navigation", "sidebar", "menu"],
        "section_list": ["section", "article", "content", "chapter"],
        "figure_gallery": ["gallery", "carousel", "swiper", "masonry"],
        "metrics_table": ["table", "metric", "kpi", "stat"],
        "timeline": ["timeline", "roadmap", "history"],
        "card_grid": ["card", "grid", "listing"],
        "footer": ["footer"],
    }

    for slot, keywords in rules.items():
        if any(word in haystack for word in keywords):
            slots.append(slot)

    if not slots:
        slots.append("generic_section")
    return slots


def line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def classify_files(
    repo_root: Path,
    max_component_files: int,
    max_style_files: int,
    max_file_size_kb: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    components: list[Path] = []
    styles: list[Path] = []
    tokens: list[Path] = []

    max_bytes = max_file_size_kb * 1024

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        base = Path(dirpath)
        for name in filenames:
            file_path = base / name
            if should_skip_file(file_path):
                continue
            if file_path.stat().st_size > max_bytes:
                continue

            lower_name = name.lower()
            ext = file_path.suffix.lower()

            if ext in COMPONENT_EXTS and len(components) < max_component_files:
                components.append(file_path)
                continue

            if ext in STYLE_EXTS and len(styles) < max_style_files:
                styles.append(file_path)
                continue

            if lower_name in TOKEN_FILE_HINTS or "token" in lower_name or "theme" in lower_name:
                tokens.append(file_path)

    components.sort(key=lambda p: (len(p.parts), p.as_posix().lower()))
    styles.sort(key=lambda p: (len(p.parts), p.as_posix().lower()))
    tokens = sorted(set(tokens), key=lambda p: (len(p.parts), p.as_posix().lower()))
    return components, styles, tokens


def copy_curated_file(src: Path, dst_root: Path, rel_path: Path) -> Path:
    dst = dst_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def as_project_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def write_manifest(
    module_repo_dir: Path,
    repo: CandidateRepo,
    default_branch: str,
    resolved_ref: str,
    source_root: Path,
    copied_components: list[dict[str, Any]],
    copied_styles: list[dict[str, Any]],
    copied_tokens: list[dict[str, Any]],
) -> Path:
    manifest = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "repo": {
            "full_name": repo.full_name,
            "html_url": repo.html_url,
            "license_spdx": repo.license_spdx,
            "license_kind": repo.license_kind,
            "final_score": round(repo.final_score, 4),
            "default_branch": default_branch,
            "resolved_ref": resolved_ref,
        },
        "source": {
            "raw_repo_root": as_project_rel(source_root),
        },
        "stats": {
            "component_count": len(copied_components),
            "style_count": len(copied_styles),
            "token_count": len(copied_tokens),
        },
        "components": copied_components,
        "styles": copied_styles,
        "tokens": copied_tokens,
    }

    manifest_path = module_repo_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def write_lock_file(
    lock_path: Path,
    source_csv: str,
    from_lock: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lock_payload = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source_csv": source_csv,
        "source_lock": from_lock,
        "selection": {
            "top_n": args.top_n,
            "min_score": args.min_score,
            "allow_restricted_license": bool(args.allow_restricted_license),
        },
        "entries": records,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local component/css module packs from ranked GitHub templates.")
    parser.add_argument(
        "--csv",
        default="",
        help="Ranked CSV path. If empty, auto-uses latest file in data/collectors/output.",
    )
    parser.add_argument(
        "--from-lock",
        default="",
        help="Load repo list from lock JSON. If set, CSV stage is skipped.",
    )
    parser.add_argument(
        "--lock-file",
        default="data/collectors/locks/template_sources.lock.json",
        help="Where to write reproducible lock JSON after extraction.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="How many repos to process.")
    parser.add_argument("--min-score", type=float, default=0.65, help="Minimum final_score to keep.")
    parser.add_argument(
        "--allow-restricted-license",
        action="store_true",
        help="Allow non-permissive/unknown licenses.",
    )
    parser.add_argument("--max-component-files", type=int, default=120)
    parser.add_argument("--max-style-files", type=int, default=80)
    parser.add_argument("--max-file-size-kb", type=int, default=300)
    parser.add_argument("--raw-dir", default="data/collectors/raw")
    parser.add_argument("--modules-dir", default="data/collectors/modules")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--request-timeout", type=float, default=25.0)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--request-backoff", type=float, default=1.2)
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN", ""),
        help="GitHub token. Defaults to env GITHUB_TOKEN or root .env.",
    )
    return parser.parse_args()


def main() -> int:
    env_path = PROJECT_ROOT / ".env"
    loaded = load_env_file(env_path)
    if loaded:
        print(f"[modules] loaded {loaded} variables from {env_path}")

    args = parse_args()
    if not args.token:
        print("[modules] warning: GITHUB_TOKEN is empty. Requests may hit rate limit.")

    output_dir = PROJECT_ROOT / "data" / "collectors" / "output"
    csv_path = ""
    from_lock_path = ""

    if args.from_lock:
        lock_path_input = Path(args.from_lock)
        lock_path_input = lock_path_input if lock_path_input.is_absolute() else PROJECT_ROOT / lock_path_input
        if not lock_path_input.exists():
            raise FileNotFoundError(f"Lock file not found: {lock_path_input}")
        candidates = read_candidates_from_lock(lock_path_input, top_n=args.top_n)
        from_lock_path = as_project_rel(lock_path_input)
        print(f"[modules] using lock: {lock_path_input}")
    else:
        resolved_csv = Path(args.csv) if args.csv else find_latest_csv(output_dir)
        resolved_csv = resolved_csv if resolved_csv.is_absolute() else PROJECT_ROOT / resolved_csv
        if not resolved_csv.exists():
            raise FileNotFoundError(f"CSV file not found: {resolved_csv}")

        candidates = read_candidates_from_csv(
            csv_path=resolved_csv,
            top_n=args.top_n,
            min_score=args.min_score,
            allow_restricted_license=args.allow_restricted_license,
        )
        csv_path = as_project_rel(resolved_csv)
        print(f"[modules] using csv: {resolved_csv}")

    print(f"[modules] selected candidates: {len(candidates)}")

    raw_dir = Path(args.raw_dir)
    modules_dir = Path(args.modules_dir)
    lock_file = Path(args.lock_file)

    raw_dir = raw_dir if raw_dir.is_absolute() else PROJECT_ROOT / raw_dir
    modules_dir = modules_dir if modules_dir.is_absolute() else PROJECT_ROOT / modules_dir
    lock_file = lock_file if lock_file.is_absolute() else PROJECT_ROOT / lock_file

    raw_dir.mkdir(parents=True, exist_ok=True)
    modules_dir.mkdir(parents=True, exist_ok=True)

    fetcher = GitHubFetcher(
        token=args.token,
        timeout=args.request_timeout,
        retries=args.request_retries,
        backoff=args.request_backoff,
    )

    index_records: list[dict[str, Any]] = []
    lock_records: list[dict[str, Any]] = []

    for idx, repo in enumerate(candidates, start=1):
        slug = repo_slug(repo.full_name)
        repo_raw_dir = raw_dir / slug
        module_repo_dir = modules_dir / slug
        module_components_dir = module_repo_dir / "components"
        module_styles_dir = module_repo_dir / "styles"
        module_tokens_dir = module_repo_dir / "tokens"

        print(f"[modules] ({idx}/{len(candidates)}) processing {repo.full_name} ...")

        source_root: Path | None = None
        default_branch = (repo.default_branch or "").strip()
        resolved_ref = (repo.preferred_ref or "").strip()

        try:
            if args.force_redownload or not repo_raw_dir.exists():
                if not default_branch:
                    default_branch = fetcher.get_default_branch(repo.full_name)

                if not resolved_ref:
                    try:
                        resolved_ref = fetcher.get_branch_head_sha(repo.full_name, default_branch)
                    except Exception:
                        resolved_ref = ""

                ref_candidates = unique_non_empty([resolved_ref, default_branch, "main", "master"])
                zip_data: bytes | None = None
                used_ref = ""
                last_error: Exception | None = None

                for ref in ref_candidates:
                    try:
                        zip_data = fetcher.download_repo_zip_by_ref(repo.full_name, ref)
                        used_ref = ref
                        break
                    except Exception as exc:
                        last_error = exc

                if zip_data is None:
                    raise RuntimeError(f"zip download failed: {last_error}")

                if not resolved_ref:
                    resolved_ref = used_ref

                source_root = extract_zip_to_dir(zip_data, repo_raw_dir)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            else:
                source_root = repo_raw_dir
                children = [p for p in source_root.iterdir() if p.is_dir()]
                if len(children) == 1:
                    source_root = children[0]

                if not default_branch:
                    default_branch = "unknown"
                if not resolved_ref:
                    resolved_ref = "local_cache_unknown"

        except Exception as exc:
            print(f"[modules] warning: skip {repo.full_name}, download failed: {exc}")
            continue

        if source_root is None or not source_root.exists():
            print(f"[modules] warning: skip {repo.full_name}, source folder missing")
            continue

        if module_repo_dir.exists():
            shutil.rmtree(module_repo_dir)
        module_components_dir.mkdir(parents=True, exist_ok=True)
        module_styles_dir.mkdir(parents=True, exist_ok=True)
        module_tokens_dir.mkdir(parents=True, exist_ok=True)

        components, styles, tokens = classify_files(
            repo_root=source_root,
            max_component_files=args.max_component_files,
            max_style_files=args.max_style_files,
            max_file_size_kb=args.max_file_size_kb,
        )

        copied_components: list[dict[str, Any]] = []
        copied_styles: list[dict[str, Any]] = []
        copied_tokens: list[dict[str, Any]] = []

        for file_path in components:
            rel = file_path.relative_to(source_root)
            copied = copy_curated_file(file_path, module_components_dir, rel)
            text = text_or_empty(file_path, max_bytes=args.max_file_size_kb * 1024)
            copied_components.append(
                {
                    "name": file_path.stem,
                    "source_path": rel.as_posix(),
                    "local_path": as_project_rel(copied),
                    "extension": file_path.suffix.lower(),
                    "lines": line_count(text),
                    "slot_candidates": infer_slot_candidates(rel, text),
                }
            )

        for file_path in styles:
            rel = file_path.relative_to(source_root)
            copied = copy_curated_file(file_path, module_styles_dir, rel)
            text = text_or_empty(file_path, max_bytes=args.max_file_size_kb * 1024)
            css_vars = len(re.findall(r"--[a-zA-Z0-9_-]+\s*:", text))
            class_selectors = len(re.findall(r"\.[a-zA-Z0-9_-]+", text))
            copied_styles.append(
                {
                    "name": file_path.stem,
                    "source_path": rel.as_posix(),
                    "local_path": as_project_rel(copied),
                    "extension": file_path.suffix.lower(),
                    "lines": line_count(text),
                    "css_variable_count": css_vars,
                    "class_selector_count": class_selectors,
                }
            )

        for file_path in tokens:
            if not file_path.exists():
                continue
            rel = file_path.relative_to(source_root)
            copied = copy_curated_file(file_path, module_tokens_dir, rel)
            copied_tokens.append(
                {
                    "name": file_path.name,
                    "source_path": rel.as_posix(),
                    "local_path": as_project_rel(copied),
                }
            )

        manifest_path = write_manifest(
            module_repo_dir=module_repo_dir,
            repo=repo,
            default_branch=default_branch,
            resolved_ref=resolved_ref,
            source_root=source_root,
            copied_components=copied_components,
            copied_styles=copied_styles,
            copied_tokens=copied_tokens,
        )

        index_records.append(
            {
                "repo": repo.full_name,
                "score": round(repo.final_score, 4),
                "license": repo.license_spdx,
                "default_branch": default_branch,
                "resolved_ref": resolved_ref,
                "module_dir": as_project_rel(module_repo_dir),
                "manifest": as_project_rel(manifest_path),
                "component_count": len(copied_components),
                "style_count": len(copied_styles),
                "token_count": len(copied_tokens),
            }
        )

        lock_records.append(
            {
                "full_name": repo.full_name,
                "html_url": repo.html_url,
                "final_score": round(repo.final_score, 4),
                "license_kind": repo.license_kind,
                "license_spdx": repo.license_spdx,
                "default_branch": default_branch,
                "resolved_ref": resolved_ref,
            }
        )

        print(
            f"[modules] done {repo.full_name}: "
            f"components={len(copied_components)}, styles={len(copied_styles)}, tokens={len(copied_tokens)}"
        )

    module_index = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source_csv": csv_path,
        "source_lock": from_lock_path,
        "repo_count": len(index_records),
        "records": index_records,
    }
    module_index_path = modules_dir / "module_index.json"
    module_index_path.write_text(json.dumps(module_index, ensure_ascii=False, indent=2), encoding="utf-8")

    write_lock_file(
        lock_path=lock_file,
        source_csv=csv_path,
        from_lock=from_lock_path,
        records=lock_records,
        args=args,
    )

    print(f"[modules] module index: {module_index_path}")
    print(f"[modules] lock file:   {lock_file}")
    print(f"[modules] completed repos: {len(index_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
