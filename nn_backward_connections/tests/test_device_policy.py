"""Tests for the device policy — the enforcement of "CUDA decides, MPS evaluates".

These run against the REAL records on disk, not fixtures, because the whole point is that
the policy holds for the actual campaign history.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from mfas.device_policy import (
    DISPERSION,
    GATING_DEVICE_CLASS,
    assert_comparable,
    device_class,
    device_config_key,
    may_gate,
    normalise_artifact_path,
)

ROOT = Path(__file__).resolve().parent.parent
CUDA_REC = "20260810T052352Z-H42-connectome-s42-implement-f41d7e"
MPS_REC = "20260825T083116Z-H42-connectome-s42-implement-f41d7e"


def _load(exp_id: str) -> dict:
    return json.loads((ROOT / "results" / f"{exp_id}.json").read_text())


def test_classifies_the_two_real_records():
    assert device_class(_load(CUDA_REC)) == "cuda"
    assert device_class(_load(MPS_REC)) == "mps"


def test_only_cuda_may_gate():
    assert may_gate(_load(CUDA_REC)) is True
    assert may_gate(_load(MPS_REC)) is False
    assert GATING_DEVICE_CLASS == "cuda"


def test_cross_device_comparison_is_refused():
    """The load-bearing rule: this delta would mostly measure hardware."""
    with pytest.raises(ValueError, match="REFUSED"):
        assert_comparable([_load(CUDA_REC), _load(MPS_REC)])


def test_same_device_comparison_is_allowed():
    assert assert_comparable([_load(CUDA_REC)]) == "cuda"


def test_unknown_device_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        device_class({"exp_id": "x"})
    with pytest.raises(ValueError):
        device_class({"exp_id": "x", "env": {"gpu": "Some Unknown Accelerator"}})


def test_the_two_records_share_a_config_hash_but_not_a_device_key():
    """Documents the defect the policy works around, so a fix cannot silently regress it."""
    cuda, mps = _load(CUDA_REC), _load(MPS_REC)
    assert cuda["config_hash"] == mps["config_hash"], "premise changed: hashes no longer collide"
    assert cuda["pct"] != mps["pct"], "premise changed: the scores no longer differ"
    assert device_config_key(cuda) != device_config_key(mps)


def test_windows_artifact_paths_normalise():
    """Half of the cross-machine problem: the recorded path must at least parse here."""
    rec = _load(CUDA_REC)
    raw = rec["best_positions_path"]
    assert "\\" in raw, "premise changed: the record no longer carries a Windows path"
    p = normalise_artifact_path(raw)
    assert p.parts[0] == "results" and p.name.endswith(".npy")


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEBT: .gitignore:30 excludes results/*_positions.npy, so the champion's "
           "ordering has never left the machine that produced it and cannot be evaluated "
           "here. tools/pin_champion.py fixes this going forward, but the CUDA champion "
           "must be pinned ON the RTX box. When that lands this test XPASSes and the "
           "marker must be removed.",
)
def test_cuda_champion_ordering_is_retained_in_the_repo():
    """The chosen workflow depends on this: the laptop must open what the RTX box wrote."""
    p = normalise_artifact_path(_load(CUDA_REC)["best_positions_path"])
    assert (ROOT / p).exists() or (ROOT / "results" / "champions" /
                                   "H42_connectome_s42_cuda.npz").exists()


def test_posix_paths_survive_normalisation():
    p = normalise_artifact_path("results/foo_positions.npy")
    assert str(p) == "results/foo_positions.npy"
    assert normalise_artifact_path(None) is None


def test_dispersion_entries_carry_provenance():
    """A gate constant without provenance is how a false 'sigma = 0' got inherited."""
    for cls, entry in DISPERSION.items():
        assert entry["n"] >= 2
        assert len(entry["records"]) == entry["n"]
        for exp_id in entry["records"]:
            assert (ROOT / "results" / f"{exp_id}.json").exists(), f"{cls}: missing {exp_id}"


def test_cuda_dispersion_claim_matches_the_records_on_disk():
    """Re-derive the 0.000000 pp claim rather than trusting the table."""
    pcts = [_load(e)["pct"] for e in DISPERSION["cuda"]["records"]]
    assert max(pcts) - min(pcts) == pytest.approx(DISPERSION["cuda"]["range_pp"], abs=1e-9)


def test_mps_dispersion_claim_matches_the_records_on_disk():
    pcts = [_load(e)["pct"] for e in DISPERSION["mps"]["records"]]
    assert max(pcts) - min(pcts) == pytest.approx(DISPERSION["mps"]["range_pp"], abs=1e-6)


def test_no_gating_device_record_is_silently_absent():
    """If any CUDA record for the champion vanishes, the policy's evidence is gone."""
    found = [f for f in glob.glob(str(ROOT / "results" / "*H42-connectome-s42-*.json"))
             if device_class(json.loads(Path(f).read_text())) == "cuda"]
    assert len(found) >= 3


# --------------------------------------------------------------- tools/pin_champion.py

def _pin_mod():
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "pin_champion", ROOT / "tools" / "pin_champion.py")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["pin_champion"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pinned_mps_champion_round_trips_through_the_oracle():
    """A pinned ordering must re-score to its recorded value exactly, or it is not that run."""
    npz = ROOT / "results" / "champions" / "H42_connectome_s42_mps.npz"
    if not npz.exists():
        pytest.skip("nothing pinned yet on this machine")
    _pin_mod().verify(npz)  # raises SystemExit on any mismatch


def test_pin_refuses_when_the_artifact_is_absent():
    """The refusal must be explicit and actionable, never a silent skip."""
    mod = _pin_mod()
    with pytest.raises(SystemExit, match="never committed|not on this machine"):
        mod.pin(ROOT / "results" / f"{CUDA_REC}.json")


def test_positions_to_rank_rejects_a_non_permutation():
    import numpy as np
    mod = _pin_mod()
    with pytest.raises(ValueError, match="permutation"):
        mod.positions_to_rank(np.zeros(10, dtype=np.int64))


def test_positions_to_rank_accepts_float_positions_and_int_ranks():
    import numpy as np
    mod = _pin_mod()
    pos = np.array([3.5, 0.1, 9.9, 2.2])
    assert mod.positions_to_rank(pos).tolist() == [2, 0, 3, 1]
    assert mod.positions_to_rank(np.array([2, 0, 3, 1])).tolist() == [2, 0, 3, 1]
