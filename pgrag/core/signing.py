"""Signing for the decisions audit trail. Pure standard library.

decisions/ is a plain folder: anyone who can write to it could otherwise hand
the pipeline a decision no reviewer made (a forged decisions_v9999.json), or
delete the newest file to bring back an older, more permissive one. So:

  - every decisions file carries an HMAC-SHA256 signature over its contents,
    made with a key kept OUTSIDE the project (~/.config/pgrag/review.key, or
    $PGRAG_KEY_DIR) - write access to decisions/ is not enough to sign;
  - every file records the signature of the version before it, so the files
    form a chain and an old version cannot be edited, inserted or removed;
  - the newest version and its signature are also recorded next to the key
    (the "head"), so deleting the newest file is caught as a rollback.

HMAC, not a public-key signature: the reviewer who signs and the pipeline that
verifies run as the same user on the same machine, and it keeps this module
free of third-party dependencies.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

KEY_DIR_ENV = "PGRAG_KEY_DIR"
KEY_NAME = "review.key"
HEADS_NAME = "heads.json"
SIGNATURE_KEY = "signature"


def key_dir() -> Path:
    """Where the key and the heads live. Never inside the project."""
    env = os.environ.get(KEY_DIR_ENV)
    return Path(env) if env else Path.home() / ".config" / "pgrag"


def load_key(create: bool = False) -> bytes | None:
    """The signing key, or None if there is none yet (and create is False)."""
    path = key_dir() / KEY_NAME
    if path.exists():
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    if not create:
        return None
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = secrets.token_bytes(32)
    # Created owner-only from the start, not chmod-ed after a readable write.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key.hex())
    return key


def _canonical(payload: dict) -> bytes:
    body = {k: v for k, v in payload.items() if k != SIGNATURE_KEY}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(payload: dict, key: bytes) -> str:
    return hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()


def verify(payload: dict, key: bytes) -> bool:
    signature = payload.get(SIGNATURE_KEY)
    return isinstance(signature, str) and hmac.compare_digest(signature, sign(payload, key))


def _heads() -> dict:
    try:
        data = json.loads((key_dir() / HEADS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_head(decisions_dir: Path) -> dict | None:
    """{"version": N, "signature": ...} last signed for this folder, or None."""
    head = _heads().get(str(decisions_dir.resolve()))
    return head if isinstance(head, dict) else None


def write_head(decisions_dir: Path, version: int, signature: str) -> None:
    heads = _heads()
    heads[str(decisions_dir.resolve())] = {"version": version, "signature": signature}
    path = key_dir() / HEADS_NAME
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(heads, indent=2), encoding="utf-8")
    tmp.replace(path)
