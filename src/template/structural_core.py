from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from src.contracts.schemas import ShellWrapperSignature

MAX_WRAPPER_CHAIN_DEPTH = 2
SHELL_CONTAINER_TAGS = {"section", "div", "article", "header", "main", "aside", "nav", "footer"}
SHELL_GENERIC_TOKENS = {
    "content",
    "container",
    "wrapper",
    "box",
    "visual",
    "section",
    "main",
}
REGION_ROLE_HINTS: dict[str, set[str]] = {
    "hero": {"hero", "lead", "intro", "header", "catchphrase", "headline"},
    "section": {"section", "content", "body", "copy", "swatch"},
    "gallery": {"gallery", "media", "figure", "carousel"},
    "table": {"table", "results", "metrics", "data"},
    "footer": {"footer", "meta", "citation"},
    "nav": {"nav", "menu", "brand", "action"},
}


def dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def tag_classes(tag: Tag) -> list[str]:
    return dedupe_strings([str(item).strip() for item in tag.get("class", [])])


def tag_ids(tag: Tag) -> list[str]:
    element_id = str(tag.get("id") or "").strip()
    return [element_id] if element_id else []


def tag_tokens(tag: Tag) -> set[str]:
    raw_parts = [str(tag.name or "")]
    raw_parts.extend(tag_classes(tag))
    raw_parts.extend(tag_ids(tag))
    tokens: set[str] = set()
    for part in raw_parts:
        for token in str(part).replace("_", "-").split("-"):
            clean = token.strip().lower()
            if clean:
                tokens.add(clean)
    return tokens


def selector_tokens(selector: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(selector or "").lower())
        if token
    }


def is_meaningful_wrapper(tag: Tag) -> bool:
    return (tag.name or "") not in {"html", "body"} and bool(tag_classes(tag) or tag_ids(tag))


def wrapper_signature_from_tag(tag: Tag) -> ShellWrapperSignature:
    return ShellWrapperSignature(
        tag=str(tag.name or "div"),
        required_classes=tag_classes(tag),
        preserve_ids=tag_ids(tag),
    )


def capture_wrapper_chain(tag: Tag, max_depth: int = MAX_WRAPPER_CHAIN_DEPTH) -> list[ShellWrapperSignature]:
    wrappers: list[ShellWrapperSignature] = []
    for ancestor in tag.parents:
        if not isinstance(ancestor, Tag):
            continue
        if not is_meaningful_wrapper(ancestor):
            continue
        wrappers.append(wrapper_signature_from_tag(ancestor))
        if len(wrappers) >= max_depth:
            break
    return wrappers


def selector_segment(tag: Tag) -> str:
    tag_name = str(tag.name or "div")
    tag_id = str(tag.get("id") or "").strip()
    if tag_id:
        return f"{tag_name}#{tag_id}"

    classes = [name for name in tag_classes(tag)[:2] if name]
    if classes:
        return tag_name + "".join(f".{name}" for name in classes)

    siblings = [sibling for sibling in tag.find_previous_siblings(tag_name) if isinstance(sibling, Tag)]
    if siblings:
        return f"{tag_name}:nth-of-type({len(siblings) + 1})"
    return tag_name


def build_unique_selector(tag: Tag, soup: BeautifulSoup) -> str:
    segments: list[str] = []
    current: Tag | None = tag
    while current is not None and isinstance(current, Tag):
        if str(current.name or "") in {"html", "body"}:
            break
        segments.insert(0, selector_segment(current))
        selector = " > ".join(segments)
        try:
            matches = [match for match in soup.select(selector) if isinstance(match, Tag)]
        except Exception:
            matches = []
        if len(matches) == 1 and matches[0] is tag:
            return selector
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    return " > ".join(segments) or str(tag.name or "div")


def select_unique_tag(soup: BeautifulSoup, selector: str) -> Tag | None:
    try:
        matches = soup.select(str(selector or ""))
    except Exception:
        return None
    tags = [match for match in matches if isinstance(match, Tag)]
    if len(tags) != 1:
        return None
    return tags[0]


def dom_index_for_tag(template_soup: BeautifulSoup, target: Tag) -> int:
    root = template_soup.body or template_soup
    for dom_index, tag in enumerate(root.find_all(True)):
        if tag is target:
            return dom_index
    return 10_000


def matches_wrapper_signature(tag: Tag, signature: ShellWrapperSignature) -> bool:
    if str(tag.name or "") != str(signature.tag or ""):
        return False
    actual_classes = set(tag_classes(tag))
    if any(required not in actual_classes for required in signature.required_classes):
        return False
    expected_ids = set(signature.preserve_ids)
    actual_ids = set(tag_ids(tag))
    if expected_ids and expected_ids != actual_ids:
        return False
    return True
