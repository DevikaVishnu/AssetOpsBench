"""Locator and hash resolver shared by the semantic adapter and its verifier.

This is the only code the adapter (``_semantic_adapter``) and the verifier
(``_semantic_verify``) share.  It resolves ``raw_ref`` locators such as
``events[seq=125].input_batches[0][46].content`` against a raw record, and
hashes content the way the capture does (``_subagent_capture._content_digest``):
strings as UTF-8, anything else as ``json.dumps(value, default=str)``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

DIGEST_BASIS = "sha256 of the UTF-8 text; non-string content as json.dumps(value, default=str), as the capture hashes it"

_STEP = re.compile(r"\.?([A-Za-z_]\w*)|\[(\w+)=([^\]]+)\]|\[(\d+)\]")


class LocatorError(LookupError):
    """A locator does not resolve to exactly one value."""


def digest_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(digest_text(value).encode("utf-8")).hexdigest()


def size(value: Any) -> int:
    """Length in code points of the digest text."""
    return len(digest_text(value))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_name(key: Any) -> bool:
    """True if ``key`` can be written as a ``.name`` locator step."""
    return isinstance(key, str) and re.fullmatch(r"[A-Za-z_]\w*", key) is not None


def _steps(locator: Any) -> list[tuple[str, str, str, str]]:
    if not isinstance(locator, str) or not locator or locator.startswith("."):
        raise LocatorError(f"malformed locator {locator!r}")
    steps, pos = [], 0
    while pos < len(locator):
        m = _STEP.match(locator, pos)
        if m is None or (m.group(1) and pos > 0 and not m.group(0).startswith(".")):
            raise LocatorError(f"malformed locator {locator!r} at {pos}")
        steps.append(m.groups(default=""))
        pos = m.end()
    return steps


def resolve(record: Any, locator: str) -> Any:
    """Resolve ``locator`` against ``record``; raise :class:`LocatorError` unless it names exactly one value."""
    obj = record
    for name, key, val, idx in _steps(locator):
        try:
            if name:
                if not isinstance(obj, dict) or name not in obj:
                    raise LocatorError(f"{locator}: no field {name!r}")
                obj = obj[name]
            elif key:
                matches = [x for x in obj if isinstance(x, dict) and str(x.get(key)) == val]
                if len(matches) != 1:
                    raise LocatorError(f"{locator}: {len(matches)} matches for [{key}={val}]")
                obj = matches[0]
            else:
                if not isinstance(obj, list):
                    raise LocatorError(f"{locator}: [{idx}] on a non-list")
                obj = obj[int(idx)]
        except (IndexError, TypeError) as exc:
            raise LocatorError(f"{locator}: {exc}") from exc
    return obj
