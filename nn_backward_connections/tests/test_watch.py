"""Tests for the campaign watchdog.

The watchdog is the only thing that converts a silently-dead run into a human-visible event,
so its classification must be exercised against every state, using a temp copy of the
autoresearch layout rather than the live campaign files.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_watch(ar_dir: Path):
    """Load watch.py with its AR/STATE/LOCK paths redirected into a temp dir."""
    spec = importlib.util.spec_from_file_location(
        "watch_under_test", ROOT / "autoresearch" / "watch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["watch_under_test"] = mod
    spec.loader.exec_module(mod)
    mod.AR = ar_dir
    mod.STATE = ar_dir / "state.json"
    mod.DRIVER_LOG = ar_dir / "logs" / "driver.log"
    mod.LOCK = ar_dir / ".lock"
    mod.STOP = ar_dir / "STOP"
    mod.ALERTS_LOG = ar_dir / "ALERTS.log"
    mod.ALERT_LATEST = ar_dir / "ALERT"
    return mod


@pytest.fixture
def ar(tmp_path):
    (tmp_path / "logs").mkdir()
    return tmp_path


def _write_state(ar: Path, **kw):
    (ar / "state.json").write_text(json.dumps({"cycle": 8, "mode": "incremental", **kw}))


def test_running_ok_when_fresh_and_locked(ar):
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    (ar / ".lock" / "cycle.started").write_text("x")
    h = w.assess(time.time(), stale_s=90 * 60)
    assert h.status == "RUNNING_OK" and h.code == 0 and h.ok


def test_blocked_mode_is_caught(ar):
    w = _load_watch(ar)
    _write_state(ar, mode="blocked", last_cycle_outcome="oracle integrity failed")
    (ar / ".lock").mkdir()
    h = w.assess(time.time(), stale_s=90 * 60)
    assert h.status == "BLOCKED" and h.code == 4
    assert "integrity" in h.detail


def test_no_driver_when_lock_absent_and_no_stop(ar):
    w = _load_watch(ar)
    _write_state(ar)
    h = w.assess(time.time(), stale_s=90 * 60)
    assert h.status == "NO_DRIVER" and h.code == 5


def test_deliberate_stop_is_not_flagged_as_dead(ar):
    """A STOP file is an intentional shutdown, not a crash -- do not alert on it."""
    w = _load_watch(ar)
    _write_state(ar)
    (ar / "STOP").write_text("")
    h = w.assess(time.time(), stale_s=90 * 60)
    assert h.status != "NO_DRIVER"


def test_stale_when_activity_is_old(ar):
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    old = time.time() - 3 * 3600
    import os
    for p in (ar / "state.json", ar / "logs" / "driver.log", ar / ".lock"):
        if not p.exists():
            p.write_text("x")
        os.utime(p, (old, old))
    h = w.assess(time.time(), stale_s=90 * 60)
    assert h.status == "STALE" and h.code == 3
    assert h.age_s > 90 * 60


def test_fresh_activity_clears_stale(ar):
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    (ar / "logs" / "driver.log").write_text("just now")
    h = w.assess(time.time(), stale_s=90 * 60)
    assert h.status == "RUNNING_OK"


def test_alert_writes_a_file_and_carries_the_nonce(ar):
    w = _load_watch(ar)
    receipt = w.raise_alert(w.Health("STALE", "test", 8, 5400.0), nonce="abc123")
    assert receipt["logged"] is True
    assert "abc123" in (ar / "ALERT").read_text()
    assert "abc123" in (ar / "ALERTS.log").read_text()


def test_alert_appends_history(ar):
    w = _load_watch(ar)
    w.raise_alert(w.Health("BLOCKED", "one", 8, None), nonce="n1")
    w.raise_alert(w.Health("STALE", "two", 8, 100.0), nonce="n2")
    hist = (ar / "ALERTS.log").read_text()
    assert "n1" in hist and "n2" in hist
    assert (ar / "ALERT").read_text().count("n2") == 1  # latest only


def test_exit_codes_are_distinct_per_status(ar):
    w = _load_watch(ar)
    assert set(w.CODES.values()) == {0, 3, 4, 5, 6}
    assert w.CODES["RUNNING_OK"] == 0
