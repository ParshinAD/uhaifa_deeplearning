"""Frozen-oracle integrity gate.

The four FROZEN files define the ground-truth metric / harness / tests:

    src/mfas/metrics.py, eval/harness.py, eval/aggregate.py, tests/test_metrics.py

The PreToolUse hook (``.claude/hooks/protect-frozen-files.sh``) blocks ``Edit``/``Write``
to them, but a ``Bash`` write (``sed -i``, ``>``, ``python -c "open(...).write()"``) bypasses
that hook. This module closes that hole at the *scoring* layer: any code path that produces or
accepts a score (``eval/run_variant.py``, used by both the implementer and the verifier) calls
:func:`verify_frozen_manifest` BEFORE doing so. On any mismatch it aborts with a clear error, so a
tampered oracle can never silently mint a "win".

This is intentionally pure-stdlib and imports NONE of the frozen modules (hashing their bytes,
not their behaviour). The reference manifest is ``eval/frozen.sha256`` (``shasum -a 256`` format:
``<hex>  <repo-relative-path>``), itself protected by the hook + chmod 0444.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

# Repo root = parent of this file's directory (eval/ -> repo root).
_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _ROOT / "eval" / "frozen.sha256"


class FrozenIntegrityError(SystemExit):
    """Raised (as SystemExit) when the frozen-file manifest does not match on disk."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest(text: str):
    """Parse ``shasum -a 256`` lines into ``[(expected_hex, rel_path), ...]``."""
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "<hex>  <path>" (two spaces, GNU/BSD shasum). Split on first run of spaces.
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise FrozenIntegrityError(
                f"FROZEN INTEGRITY ABORT: malformed manifest line: {raw!r}")
        entries.append((parts[0].lower(), parts[1].strip()))
    return entries


def verify_frozen_manifest(manifest: Optional[Path] = None) -> None:
    """Abort (SystemExit) unless every file in the manifest matches its recorded SHA-256.

    Called before any score is produced or accepted. Raises :class:`FrozenIntegrityError`
    (a ``SystemExit`` subclass) on a missing/empty manifest, a missing file, or a hash mismatch.
    Returns ``None`` on success.
    """
    man_path = Path(manifest) if manifest is not None else _MANIFEST
    if not man_path.exists():
        raise FrozenIntegrityError(
            f"FROZEN INTEGRITY ABORT: manifest not found at {man_path}. "
            f"Refusing to score without the integrity reference.")

    entries = _parse_manifest(man_path.read_text())
    if not entries:
        raise FrozenIntegrityError(
            f"FROZEN INTEGRITY ABORT: manifest {man_path} is empty.")

    mismatches = []
    for expected, rel in entries:
        fpath = (_ROOT / rel).resolve()
        if not fpath.exists():
            mismatches.append(f"  MISSING   {rel}")
            continue
        actual = _sha256(fpath)
        if actual != expected:
            mismatches.append(f"  MODIFIED  {rel}\n      expected {expected}\n      actual   {actual}")

    if mismatches:
        raise FrozenIntegrityError(
            "FROZEN INTEGRITY ABORT: the frozen oracle/harness/tests have been altered.\n"
            "A score cannot be trusted while these differ from eval/frozen.sha256:\n"
            + "\n".join(mismatches)
            + "\nRestore the original frozen files (git checkout) before running. See CLAUDE.md.")


if __name__ == "__main__":  # pragma: no cover - manual check
    verify_frozen_manifest()
    print("frozen integrity OK")
