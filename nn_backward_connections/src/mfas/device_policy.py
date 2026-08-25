"""Device policy: which machine may decide, and what may be compared with what.

THE DECISION (researcher, 2026-08-25): **CUDA is the canonical execution device.**
Almost everything runs on the Windows/RTX 4060 box. The Apple MPS laptop is used to
EVALUATE results after an autonomous run, never to produce a number that decides a gate.

This module exists so that decision cannot be violated by accident. It is not prose.

WHY, measured on this repo (same variant H42, same dataset, same seed 42, same commit):

    CUDA (RTX 4060)   n=3   range = 0.000000 pp   -> bit-identical, genuinely deterministic
    Apple MPS         n=2   range = 0.009983 pp   -> NOT deterministic

    connectome minimum effect size (screen gate) = 0.012 pp
    MPS same-seed range / gate                   = 0.83x
    MPS-vs-CUDA offset (84.13016 vs 84.15410)    = 0.0239 pp = 2.0x the gate

Two consequences follow, and both are enforced below rather than remembered:

1. **Cross-device deltas are REFUSED, never corrected.** The offset is twice the gate, so
   an "advantage" measured on MPS against a CUDA champion is mostly a measurement of the
   laptop. There is no correction factor that fixes this; a subtraction would just launder
   an unmeasured bias into a result.

2. **The one-seed screen policy is CUDA-only.** `autoresearch/seed_class.py` may declare a
   variant deterministic and run a single seed on the primaries. That is justified by the
   n=3 CUDA measurement above. On MPS the same claim is false: one run carries about
   +/-0.005 pp of pure device noise against a 0.012 pp gate.

A NOTE ON WHAT IS STILL SAFE ON THE LAPTOP. The non-determinism is in PRODUCING an order,
not in scoring one. Re-scoring a stored order through the frozen oracle is exact, integer,
CPU-only and device-independent, so post-hoc evaluation -- re-scoring, residual diagnostics,
probes, tiny-n theory work -- is fully valid on MPS. That is exactly the workflow chosen.
"""
from __future__ import annotations

from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, Mapping, Optional

__all__ = [
    "GATING_DEVICE_CLASS",
    "DISPERSION",
    "device_class",
    "may_gate",
    "assert_comparable",
    "normalise_artifact_path",
    "device_config_key",
]

#: The only device class whose runs may decide a gate. See the module docstring.
GATING_DEVICE_CLASS = "cuda"

#: Measured same-seed dispersion per device class. Every entry carries its provenance;
#: an unmeasured device class is absent, never assumed to be zero.
DISPERSION: Dict[str, Dict[str, Any]] = {
    "cuda": {
        "range_pp": 0.0,
        "n": 3,
        "variant": "H42",
        "dataset": "connectome",
        "seed": 42,
        "records": [
            "20260810T052352Z-H42-connectome-s42-implement-f41d7e",
            "20260810T064240Z-H42-connectome-s42-confirm-f41d7e",
            "20260810T214020Z-H42-connectome-s42-verify-f41d7e",
        ],
        "note": "bit-identical; init comes from deterministic greedy-FAS so the RNG is never drawn",
    },
    "mps": {
        "range_pp": 0.009983,
        "n": 2,
        "variant": "H42",
        "dataset": "connectome",
        "seed": 42,
        "records": [
            "20260825T083116Z-H42-connectome-s42-implement-f41d7e",
            "20260825T084524Z-H42-connectome-s42-verify-f41d7e",
        ],
        "note": "PROVISIONAL: n=2 is a range, not a sigma. Not a gating device, so not on the "
                "critical path; retained as the evidence for the policy, not as a gate constant.",
    },
}


def device_class(record: Mapping[str, Any]) -> str:
    """Classify a run record's execution device as ``cuda`` / ``mps`` / ``cpu``.

    Derived from ``env``, which every record carries. Raises rather than guessing: a
    record whose device cannot be established must not silently join a comparison.
    """
    env = record.get("env")
    if not isinstance(env, Mapping):
        raise ValueError(f"record {record.get('exp_id', '?')!r} has no env block; device unknown")
    if env.get("cuda"):
        return "cuda"
    gpu = str(env.get("gpu") or "")
    if "MPS" in gpu.upper():
        return "mps"
    if gpu in ("", "None", "cpu", "CPU"):
        return "cpu"
    raise ValueError(
        f"record {record.get('exp_id', '?')!r}: cannot classify device from env={dict(env)!r}"
    )


def may_gate(record: Mapping[str, Any]) -> bool:
    """True iff this run is allowed to decide a screen/confirm/promotion gate."""
    return device_class(record) == GATING_DEVICE_CLASS


def assert_comparable(records: Iterable[Mapping[str, Any]]) -> str:
    """Refuse a comparison that mixes device classes. Returns the common class.

    There is deliberately no ``correct_for_device`` counterpart. The measured offset is
    twice the decision gate and rests on n=2; subtracting it would convert an unmeasured
    bias into an apparently clean number, which is the failure this module exists to stop.
    """
    seen = {}
    for r in records:
        seen.setdefault(device_class(r), []).append(r.get("exp_id", "?"))
    if not seen:
        raise ValueError("no records to compare")
    if len(seen) > 1:
        detail = "; ".join(f"{k}: {', '.join(v[:3])}" for k, v in sorted(seen.items()))
        raise ValueError(
            "REFUSED: cross-device comparison. The measured MPS-vs-CUDA offset is 0.0239 pp "
            f"= 2.0x the connectome gate, so this delta would mostly measure hardware. [{detail}]"
        )
    return next(iter(seen))


def normalise_artifact_path(path: Optional[str]) -> Optional[PurePath]:
    """Make a recorded artifact path readable on either machine.

    157 of 158 campaign records were written on Windows and carry backslash paths
    (``results\\...npy``). Read literally on macOS that is a single filename containing
    backslashes and the artifact is unreachable -- which would break the very workflow
    chosen here, where the laptop evaluates artifacts the RTX box produced.
    """
    if not path:
        return None
    text = str(path)
    if "\\" in text and "/" not in text:
        return PurePosixPath(*PureWindowsPath(text).parts)
    return PurePosixPath(text)


def device_config_key(record: Mapping[str, Any]) -> str:
    """The grouping key a config hash SHOULD have included.

    Both the CUDA and the MPS H42 runs carry ``config_hash = f41d7ef84c12`` while scoring
    84.154095 and 84.125170, so the existing hash pools runs that must never pool. This
    returns the key to group on until the hash itself is repaired at its (unfrozen) caller.
    """
    return f"{record.get('algo', '?')}|{record.get('dataset', '?')}|{device_class(record)}"
