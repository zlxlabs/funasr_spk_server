"""Stable ASR capability contract shared by HTTP and WebSocket responses."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def build_asr_capabilities(
    *,
    schema_version: int,
    engine: str,
    runtime: str,
    features: Mapping[str, bool],
) -> dict[str, object]:
    """Build the minimal capability contract from explicit, non-secret inputs.

    ``capability_id`` hashes only the four protocol fields, so connection state,
    readiness and machine-specific diagnostics cannot change the identifier.
    """
    payload = {
        "schema_version": schema_version,
        "engine": engine,
        "runtime": runtime,
        "features": {"terms": bool(features.get("terms", False))},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "capability_id": hashlib.sha256(canonical.encode()).hexdigest()}
