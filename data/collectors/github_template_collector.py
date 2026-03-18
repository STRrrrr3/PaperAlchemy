#!/usr/bin/env python3
"""
Collect and rank GitHub repositories as reusable front-end templates.

This script uses GitHub REST API search and readme endpoints, then exports
ranked results into CSV and JSON for downstream Planner/Coder usage.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE = "https://api.github.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "mpl-2.0",
}

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

        # Allow quoted strings in .env
        if len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value
            loaded += 1

    return loaded

@dataclass
class RepoScore:
    full_name: str
    html_url: str
    description: str
    language: str
    stars: int
    forks: int
    pushed_at: str
    days_since_push: int
    license_spdx: str
    license_kind: str
    is_template: bool
    homepage: str
    has_demo_hint: bool
    readme_chars: int
    topics: str
    activity_score: float
    popularity_score: float
    license_score: float
    reuse_score: float
    final_score: float


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        timeout: float = 20.0,
        max_retries: int = 4,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 30.0,
    ) -> None:
        self.token = token or ""
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.1, float(retry_backoff))
        self.max_retry_delay = max(1.0, float(max_retry_delay))

    def _compute_retry_delay(
        self, attempt: int, retry_after: str | None = None, reset_at: str | None = None
    ) -> float:
        if retry_after:
            try:
                delay = float(retry_after)
                if delay > 0:
                    return min(delay, self.max_retry_delay)
            except Exception:
                pass

        if reset_at:
            try:
                reset_epoch = int(reset_at)
                now_epoch = int(time.time())
                delay = reset_epoch - now_epoch + 1
                if delay > 0:
                    return min(float(delay), self.max_retry_delay)
            except Exception:
                pass

        delay = self.retry_backoff * (2 ** max(attempt - 1, 0))
        if delay < 0.5:
            delay = 0.5
        if delay > self.max_retry_delay:
            delay = self.max_retry_delay
        return delay

    def request_json(
        self, path: str, params: dict[str, Any] | None = None, accept: str | None = None
    ) -> tuple[Any, dict[str, str]]:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{API_BASE}{path}{query}"
        headers = {
            "Accept": accept or "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "PaperAlchemy-Template-Collector",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = Request(url=url, headers=headers, method="GET")
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                    meta = {k.lower(): v for k, v in resp.headers.items()}
                    return json.loads(body), meta
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                meta = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
                status = int(exc.code)
                rate_limited = status == 403 and str(meta.get("x-ratelimit-remaining", "")) == "0"
                transient = status in (429, 500, 502, 503, 504) or rate_limited

                if transient and attempt < self.max_retries:
                    delay = self._compute_retry_delay(
                        attempt,
                        retry_after=meta.get("retry-after"),
                        reset_at=meta.get("x-ratelimit-reset"),
                    )
                    print(
                        f"[collector] warning: HTTP {status} for {url}. "
                        f"retry {attempt}/{self.max_retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                msg = f"HTTP {status} for {url}"
                if body:
                    msg += f" | {body[:300]}"
                raise RuntimeError(msg) from exc
            except (URLError, ConnectionResetError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = self._compute_retry_delay(attempt)
                    print(
                        f"[collector] warning: network error for {url}: {exc}. "
                        f"retry {attempt}/{self.max_retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Network error for {url}: {exc}") from exc

        raise RuntimeError(f"Request failed for {url}: {last_error}")

    def search_repositories(
        self, query: str, sort: str, order: str, per_page: int, page: int
    ) -> tuple[dict[str, Any], dict[str, str]]:
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page,
            "page": page,
        }
        payload, headers = self.request_json("/search/repositories", params=params)
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected response shape from search endpoint.")
        return payload, headers

    def get_readme_text(self, full_name: str) -> str:
        try:
            payload, _ = self.request_json(f"/repos/{full_name}/readme")
        except RuntimeError as exc:
            message = str(exc)
            if "HTTP 404" in message:
                return ""
            print(f"[collector] warning: readme fetch failed for {full_name}: {message}")
            return ""

        if not isinstance(payload, dict):
            return ""
        encoded = payload.get("content")
        if not encoded:
            return ""

        try:
            raw = base64.b64decode(encoded, validate=False)
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def to_days_since_push(pushed_at: str) -> int:
    if not pushed_at:
        return 9999
    try:
        pushed_time = dt.datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        now = dt.datetime.now(dt.timezone.utc)
        return max((now - pushed_time).days, 0)
    except Exception:
        return 9999


def classify_license(spdx_id: str) -> tuple[str, float]:
    if not spdx_id:
        return "missing", 0.0
    spdx = spdx_id.lower()
    if spdx in PERMISSIVE_LICENSES:
        return "permissive", 1.0
    return "restricted_or_unknown", 0.4


def readme_quality(readme_text: str) -> tuple[float, bool]:
    if not readme_text:
        return 0.1, False

    text = readme_text.lower()
    chars = len(readme_text)
    size_score = 0.2
    if chars >= 3000:
        size_score = 0.9
    elif chars >= 1500:
        size_score = 0.7
    elif chars >= 700:
        size_score = 0.5

    keyword_bonus = 0.0
    for word in ("install", "usage", "example", "license"):
        if word in text:
            keyword_bonus += 0.05

    has_demo_hint = any(k in text for k in ("demo", "preview", "screenshot", "storybook"))
    if has_demo_hint:
        keyword_bonus += 0.1

    return clamp(size_score + keyword_bonus, 0.0, 1.0), has_demo_hint


def score_repositories(
    repos: list[dict[str, Any]],
    include_non_permissive: bool,
    with_readme: bool,
    client: GitHubClient,
    sleep_seconds: float,
) -> list[RepoScore]:
    if not repos:
        return []

    max_stars = max(safe_int(r.get("stargazers_count")) for r in repos) or 1
    max_forks = max(safe_int(r.get("forks_count")) for r in repos) or 1

    scored: list[RepoScore] = []
    for idx, repo in enumerate(repos, start=1):
        full_name = str(repo.get("full_name", ""))
        if not full_name:
            continue

        stars = safe_int(repo.get("stargazers_count"))
        forks = safe_int(repo.get("forks_count"))
        pushed_at = str(repo.get("pushed_at", ""))
        days_since_push = to_days_since_push(pushed_at)

        license_obj = repo.get("license") or {}
        spdx = str(license_obj.get("spdx_id") or "")
        license_kind, license_score = classify_license(spdx)
        if license_kind != "permissive" and not include_non_permissive:
            continue

        readme_text = ""
        if with_readme:
            readme_text = client.get_readme_text(full_name)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        readme_score, has_demo_from_readme = readme_quality(readme_text)

        has_homepage = bool(str(repo.get("homepage") or "").strip())
        is_template = bool(repo.get("is_template"))
        has_demo_hint = has_homepage or has_demo_from_readme

        star_norm = math.log10(stars + 1) / math.log10(max_stars + 1)
        fork_norm = math.log10(forks + 1) / math.log10(max_forks + 1)
        popularity_score = clamp(0.75 * star_norm + 0.25 * fork_norm, 0.0, 1.0)

        # Full score at <=14 days, linearly decays to zero at 540 days.
        if days_since_push <= 14:
            activity_score = 1.0
        else:
            activity_score = clamp(1.0 - (days_since_push - 14) / (540 - 14), 0.0, 1.0)

        reuse_score = 0.0
        reuse_score += 0.35 if is_template else 0.1
        reuse_score += 0.25 if has_demo_hint else 0.05
        reuse_score += 0.20 if len(repo.get("topics") or []) >= 3 else 0.1
        reuse_score += 0.20 * readme_score
        reuse_score = clamp(reuse_score, 0.0, 1.0)

        final_score = (
            0.30 * activity_score
            + 0.30 * popularity_score
            + 0.20 * license_score
            + 0.20 * reuse_score
        )

        scored.append(
            RepoScore(
                full_name=full_name,
                html_url=str(repo.get("html_url", "")),
                description=str(repo.get("description") or "").replace("\n", " ").strip(),
                language=str(repo.get("language") or ""),
                stars=stars,
                forks=forks,
                pushed_at=pushed_at,
                days_since_push=days_since_push,
                license_spdx=spdx,
                license_kind=license_kind,
                is_template=is_template,
                homepage=str(repo.get("homepage") or ""),
                has_demo_hint=has_demo_hint,
                readme_chars=len(readme_text),
                topics=",".join(repo.get("topics") or []),
                activity_score=round(activity_score, 4),
                popularity_score=round(popularity_score, 4),
                license_score=round(license_score, 4),
                reuse_score=round(reuse_score, 4),
                final_score=round(final_score, 4),
            )
        )

        if idx % 20 == 0:
            print(f"[collector] scored {idx}/{len(repos)} repositories...")

    scored.sort(key=lambda x: x.final_score, reverse=True)
    return scored


def to_dict_rows(items: list[RepoScore]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        row = {
            "rank": rank,
            "full_name": item.full_name,
            "html_url": item.html_url,
            "description": item.description,
            "language": item.language,
            "stars": item.stars,
            "forks": item.forks,
            "pushed_at": item.pushed_at,
            "days_since_push": item.days_since_push,
            "license_spdx": item.license_spdx,
            "license_kind": item.license_kind,
            "is_template": item.is_template,
            "homepage": item.homepage,
            "has_demo_hint": item.has_demo_hint,
            "readme_chars": item.readme_chars,
            "topics": item.topics,
            "activity_score": item.activity_score,
            "popularity_score": item.popularity_score,
            "license_score": item.license_score,
            "reuse_score": item.reuse_score,
            "final_score": item.final_score,
        }
        rows.append(row)
    return rows


def export_results(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"templates_ranked_{stamp}.csv"
    json_path = out_dir / f"templates_ranked_{stamp}.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "full_name", "final_score"])

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def collect_candidates(
    client: GitHubClient,
    query: str,
    sort: str,
    order: str,
    per_page: int,
    max_results: int,
    min_stars: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    per_page = clamp(float(per_page), 1, 100)
    per_page = int(per_page)
    max_results = min(max_results, 1000)
    page = 1
    collected: list[dict[str, Any]] = []

    while len(collected) < max_results:
        payload, headers = client.search_repositories(
            query=query, sort=sort, order=order, per_page=per_page, page=page
        )
        items = payload.get("items", [])
        if not items:
            break

        for item in items:
            if safe_int(item.get("stargazers_count")) < min_stars:
                continue
            collected.append(item)
            if len(collected) >= max_results:
                break

        remaining = headers.get("x-ratelimit-remaining")
        reset_at = headers.get("x-ratelimit-reset")
        print(
            f"[collector] page={page}, total_collected={len(collected)}, "
            f"rate_remaining={remaining}, rate_reset={reset_at}"
        )

        page += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return collected


def default_query() -> str:
    cutoff = (dt.date.today() - dt.timedelta(days=365)).isoformat()
    return (
        "template language:TypeScript stars:>30 "
        f"pushed:>{cutoff} archived:false"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GitHub repositories as UI templates.")
    parser.add_argument("--query", default=default_query(), help="GitHub search query.")
    parser.add_argument("--sort", default="stars", choices=["stars", "forks", "updated"])
    parser.add_argument("--order", default="desc", choices=["desc", "asc"])
    parser.add_argument("--max-results", type=int, default=200, help="Max items to fetch (<=1000).")
    parser.add_argument("--per-page", type=int, default=50, help="Search page size (<=100).")
    parser.add_argument("--min-stars", type=int, default=30, help="Discard repos below this star count.")
    parser.add_argument(
        "--include-non-permissive",
        action="store_true",
        help="Include repositories with non-permissive or unknown license.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip readme enrichment to reduce API requests.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between API requests to reduce throttling risk.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/collectors/output",
        help="Directory to store ranked CSV and JSON outputs.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=20.0,
        help="Single request timeout in seconds.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=4,
        help="Max retries for transient HTTP/network errors.",
    )
    parser.add_argument(
        "--request-backoff",
        type=float,
        default=1.0,
        help="Base exponential backoff seconds for retries.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN", ""),
        help="GitHub token. Defaults to env GITHUB_TOKEN.",
    )
    return parser.parse_args()


def main() -> int:
    env_path = PROJECT_ROOT / ".env"
    loaded = load_env_file(env_path)
    if loaded:
        print(f"[collector] loaded {loaded} variables from {env_path}")

    args = parse_args()
    if not args.token:
        print("[collector] warning: GITHUB_TOKEN is empty. Rate limit may be very low.")

    client = GitHubClient(
        token=args.token,
        timeout=args.request_timeout,
        max_retries=args.request_retries,
        retry_backoff=args.request_backoff,
    )
    print(f"[collector] query: {args.query}")
    repos = collect_candidates(
        client=client,
        query=args.query,
        sort=args.sort,
        order=args.order,
        per_page=args.per_page,
        max_results=args.max_results,
        min_stars=args.min_stars,
        sleep_seconds=args.sleep_seconds,
    )
    print(f"[collector] fetched candidates: {len(repos)}")

    scored = score_repositories(
        repos=repos,
        include_non_permissive=args.include_non_permissive,
        with_readme=not args.skip_readme,
        client=client,
        sleep_seconds=args.sleep_seconds,
    )
    rows = to_dict_rows(scored)
    csv_path, json_path = export_results(rows, out_dir=Path(args.out_dir))

    print(f"[collector] ranked items: {len(rows)}")
    print(f"[collector] csv:  {csv_path}")
    print(f"[collector] json: {json_path}")
    if rows:
        print("[collector] top 5:")
        for row in rows[:5]:
            print(
                f"  #{row['rank']:>2} {row['full_name']} | "
                f"score={row['final_score']} | stars={row['stars']} | "
                f"license={row['license_spdx'] or 'NONE'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







