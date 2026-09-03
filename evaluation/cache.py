"""
Content-hash file cache for LLM calls.

Keyed by sha256 of the sorted-JSON payload, so identical inputs (question,
model, prompt version, ...) hit the cache and cost nothing on reruns.
Stored as one JSON file per entry under evaluation/.cache/<namespace>/.
"""
import hashlib
import json
import time
from pathlib import Path

CACHE_ROOT = Path(__file__).resolve().parent / ".cache"


def cache_key(namespace: str, payload: dict) -> str:
    canonical = json.dumps({"ns": namespace, **payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entry_path(namespace: str, key: str) -> Path:
    return CACHE_ROOT / namespace / f"{key}.json"


def cache_get(namespace: str, payload: dict):
    """Return the cached value for this payload, or None on a miss."""
    path = _entry_path(namespace, cache_key(namespace, payload))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["value"]
    except (json.JSONDecodeError, KeyError):
        return None


def cache_put(namespace: str, payload: dict, value) -> None:
    path = _entry_path(namespace, cache_key(namespace, payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"key_inputs": payload, "value": value, "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False))


def cached(namespace: str, payload: dict, fn, use_cache: bool = True):
    """Return fn() with read-through caching.

    use_cache=False bypasses the read but still writes, so a --no-cache
    run refreshes the cache instead of abandoning it.
    Returns (value, hit: bool).
    """
    if use_cache:
        value = cache_get(namespace, payload)
        if value is not None:
            return value, True
    value = fn()
    cache_put(namespace, payload, value)
    return value, False
