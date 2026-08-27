"""Tests for the campaign watchdog.

The watchdog is the only thing that converts a silently-dead run into a human-visible event,
so its classification must be exercised against every state, using a temp copy of the
autoresearch layout rather than the live campaign files.
"""
from __future__ import annotations

import importlib.util
import json
import os
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
    # Redirect the runner-output signals too, or every test sees the LIVE repo's results/
    # directory as fresh activity and no STALE case can ever fire.
    mod.ROOT = ar_dir
    mod.WORK_DIRS = (ar_dir / "results", ar_dir / "outputs")
    mod.SWEEP_LOG = ar_dir / ".sweep" / "log"
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


# ── the driver's own budget nap is not a hang (2026-08-26) ───────────────────────────────
# driver.sh:578 parks on the rolling 24 h cap with sleep_interruptible and writes nothing for
# as long as it sleeps -- 13 h in the observed case. _newest_activity cannot tell that from a
# hang, so the watchdog paged STALE every 5 min for the whole nap (~85 alerts). The driver
# does announce its resume time before sleeping; these pin that the watchdog reads it.

def _park(ar: Path, resume_epoch: float, *, trailing: str = "") -> None:
    """Write a driver.log whose last line is the budget-wait announcement, aged out of STALE."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(resume_epoch))
    text = (
        "2026-08-26 03:04:40  budget window full: 11h15m spent in the last 24 h\n"
        "2026-08-26 03:04:40    > cap 12h. Waiting 13h05m for the window to free up "
        "(until " + stamp + ").\n"
    )
    log = ar / "logs" / "driver.log"
    log.write_text(text + trailing)
    old = time.time() - 8 * 3600
    for f in (log, ar / "state.json", ar / ".lock"):
        if f.exists():
            os.utime(f, (old, old))


def test_budget_wait_is_healthy_not_stale(ar):
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    _park(ar, time.time() + 3600)
    h = w.assess(time.time(), stale_s=360 * 60)
    assert h.status == "WAITING_BUDGET" and h.code == 0 and h.ok
    assert "budget" in h.detail


def test_budget_wait_expires_back_into_stale(ar):
    """The deadline is respected: a driver that oversleeps its own resume time is still caught."""
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    _park(ar, time.time() - 60)
    h = w.assess(time.time(), stale_s=360 * 60)
    assert h.status == "STALE" and h.code == 3


def test_any_line_after_the_wait_announcement_ends_the_wait(ar):
    """Only the LAST driver.log line counts, so real activity cancels the exemption."""
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    _park(ar, time.time() + 3600,
          trailing="2026-08-26 16:10:27  --- driver cycle #7 starting\n")
    h = w.assess(time.time(), stale_s=360 * 60)
    assert h.status == "STALE" and h.code == 3


# ── the popup channel is opt-out, the watchdog is not (2026-08-26) ───────────────────────
# The operator asked for no desktop popups. Silencing a channel must never silence the
# watchdog: ALERTS.log, the ALERT file, the webhook and above all the exit code are what a
# supervisor reads, and none of them may depend on whether a toast was drawn.

class _StubSubprocess:
    def __init__(self):
        self.calls = []

    def run(self, *a, **kw):
        self.calls.append(a)


def test_popup_optout_silences_only_the_popup(ar, monkeypatch):
    w = _load_watch(ar)
    stub = _StubSubprocess()
    monkeypatch.setattr(w, "subprocess", stub)
    monkeypatch.setenv("MFAS_NO_POPUP", "1")

    h = w.Health("STALE", "pretend hang", 7, 9999.0)
    receipt = w.raise_alert(h)

    assert stub.calls == [], "MFAS_NO_POPUP must suppress the desktop notification"
    assert receipt["logged"] is True, "the durable channels must still fire"
    assert (ar / "ALERTS.log").exists() and (ar / "ALERT").exists()
    assert h.code == 3, "the supervisor signal is the exit code and it is unaffected"


def test_popup_fires_when_the_optout_is_absent(ar, monkeypatch):
    """Guards the guard: if this ever stops firing, the opt-out test above proves nothing."""
    w = _load_watch(ar)
    stub = _StubSubprocess()
    monkeypatch.setattr(w, "subprocess", stub)
    monkeypatch.delenv("MFAS_NO_POPUP", raising=False)
    monkeypatch.setattr(w.platform, "system", lambda: "Windows")

    w.raise_alert(w.Health("STALE", "pretend hang", 7, 9999.0))
    assert stub.calls, "with no opt-out the desktop channel must still be attempted"


def test_popup_optout_treats_zero_and_empty_as_off(ar, monkeypatch):
    w = _load_watch(ar)
    stub = _StubSubprocess()
    monkeypatch.setattr(w, "subprocess", stub)
    monkeypatch.setattr(w.platform, "system", lambda: "Windows")
    for value in ("", "0", "false"):
        stub.calls.clear()
        monkeypatch.setenv("MFAS_NO_POPUP", value)
        w.raise_alert(w.Health("STALE", "pretend hang", 7, 9999.0))
        assert stub.calls, f"MFAS_NO_POPUP={value!r} must NOT count as opting out"


def test_runner_output_counts_as_activity(ar):
    """The 2026-08-27 false positive: a confirm sweep writes results/*.json for hours and
    touches nothing the watchdog used to read, so it paged STALE at 451 min on a healthy run."""
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    old = time.time() - 8 * 3600
    for f in (ar / "state.json", ar / "logs" / "driver.log"):
        if not f.exists():
            f.write_text("x")
        os.utime(f, (old, old))
    os.utime(ar / ".lock", (old, old))
    assert w.assess(time.time(), stale_s=360 * 60).status == "STALE"

    (ar / "results").mkdir()                     # the runner writes a record -> dir mtime moves
    (ar / "results" / "run.json").write_text("{}")
    assert w.assess(time.time(), stale_s=360 * 60).status == "RUNNING_OK"


def test_a_truly_dead_run_is_still_caught(ar):
    """The widened signal must not make STALE unreachable: age out the work dirs too."""
    w = _load_watch(ar)
    _write_state(ar)
    (ar / ".lock").mkdir()
    (ar / "results").mkdir()
    (ar / "results" / "run.json").write_text("{}")
    old = time.time() - 8 * 3600
    for f in (ar / "state.json", ar / "logs" / "driver.log", ar / ".lock",
              ar / "results", ar / "results" / "run.json"):
        if not f.exists():
            f.write_text("x")
        os.utime(f, (old, old))
    assert w.assess(time.time(), stale_s=360 * 60).status == "STALE"
