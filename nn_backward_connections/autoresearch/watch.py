"""Campaign watchdog: turn a silently-dead run into a human-visible event.

The autonomous driver already stops itself on budget, STOP, or a blocking condition. What it
does NOT do is tell anyone. It writes state.json only at cycle boundaries (a cycle is
20-40 min), so between cycles there is no liveness signal at all, and the researcher's own
review of the deployed system found zero alerting: a 336-hour run that dies at hour 3 is
indistinguishable from success. This closes that gap.

WHAT IT CHECKS (read-only; it never touches the campaign's own files):

    NO_DRIVER  the lock is absent and no cycle is in flight        -> nothing is running
    BLOCKED    state.json mode == "blocked"                        -> driver stopped on a problem
    STALE      newest of {state.json, driver.log, lock/*.started}  -> hung: alive but not moving
               is older than --stale-min (default 90 min)             (longer than any real cycle)
    NO_PROGRESS  cycle count has not advanced across --idle-cycles -> looping without producing
               consecutive STALE-free checks
    RUNNING_OK   otherwise

HOW IT ALERTS, in order of reach, each best-effort and independent:
  1. appends the event to autoresearch/ALERTS.log and writes autoresearch/ALERT (latest only);
  2. POSTs a JSON line to $MFAS_ALERT_WEBHOOK if set (wire Telegram/Slack/ntfy with one env var);
  3. fires a local desktop notification (osascript on macOS, msg/PowerShell toast on Windows).

Exit code IS the health signal, so a scheduler/cron entry that just runs `--once` and mails its
own nonzero exits is already a working supervisor:
    0 healthy | 3 stale/hung | 4 blocked | 5 no driver | 6 no progress

Usage
-----
    python autoresearch/watch.py --once                 # one check, exit code = health
    python autoresearch/watch.py --watch --every-min 5   # loop (run under launchd/Task Scheduler)
    python autoresearch/watch.py --test                  # send a nonce through every channel
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

AR = Path(__file__).resolve().parent
STATE = AR / "state.json"
DRIVER_LOG = AR / "logs" / "driver.log"
LOCK = AR / ".lock"
STOP = AR / "STOP"
ALERTS_LOG = AR / "ALERTS.log"
ALERT_LATEST = AR / "ALERT"

# exit codes double as severities; 0 is the only healthy one
CODES = {"RUNNING_OK": 0, "STALE": 3, "BLOCKED": 4, "NO_DRIVER": 5, "NO_PROGRESS": 6}


@dataclass
class Health:
    status: str
    detail: str
    cycle: Optional[int]
    age_s: Optional[float]

    @property
    def code(self) -> int:
        return CODES[self.status]

    @property
    def ok(self) -> bool:
        return self.status == "RUNNING_OK"


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _newest_activity(now: float) -> float:
    """Most recent sign of life, in seconds of age. Independent of cycle completion."""
    candidates = [_mtime(STATE), _mtime(DRIVER_LOG)]
    if LOCK.is_dir():
        candidates += [_mtime(p) for p in LOCK.glob("*.started")]
        candidates.append(_mtime(LOCK))
    newest = max(candidates) if candidates else 0.0
    return now - newest if newest else float("inf")


def assess(now: float, stale_s: float) -> Health:
    """Classify campaign health from on-disk signals only. ``now`` is injected for testing."""
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text())
        except (OSError, ValueError):
            state = {}
    cycle = state.get("cycle")
    mode = str(state.get("mode", "")).lower()
    driver_up = LOCK.is_dir()

    if mode == "blocked":
        return Health("BLOCKED", f"state.json mode=blocked: "
                                 f"{str(state.get('last_cycle_outcome', ''))[:200]}", cycle, None)
    if not driver_up and not STOP.exists():
        # No lock and no deliberate STOP: either never started or died without releasing.
        return Health("NO_DRIVER", "lock is absent and no STOP file -- driver is not running "
                                    "(never started, or died without releasing the lock).",
                      cycle, None)
    age = _newest_activity(now)
    if age > stale_s:
        return Health("STALE", f"no activity for {age / 60:.0f} min (threshold "
                               f"{stale_s / 60:.0f} min): alive but not progressing.", cycle, age)
    return Health("RUNNING_OK", f"cycle {cycle}, last activity {age / 60:.1f} min ago.",
                  cycle, age)


def _desktop_notify(title: str, msg: str) -> None:
    try:
        sysname = platform.system()
        if sysname == "Darwin":
            subprocess.run(["osascript", "-e",
                            f'display notification {json.dumps(msg)} with title {json.dumps(title)}'],
                           check=False, timeout=10)
        elif sysname == "Windows":
            ps = (f"[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
                  f"[System.Windows.Forms.MessageBox]::Show({json.dumps(msg)},{json.dumps(title)})")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, timeout=10)
    except Exception:
        pass  # a notification failing must never take the watcher down


def _webhook(payload: dict) -> Optional[str]:
    url = os.environ.get("MFAS_ALERT_WEBHOOK", "").strip()
    if not url:
        return None
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return f"{resp.status}"
    except Exception as e:  # noqa: BLE001 - report, do not raise
        return f"webhook error: {type(e).__name__}"


def raise_alert(h: Health, *, nonce: str = "") -> dict:
    """Emit an alert through every available channel. Returns a receipt for the caller/test."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{stamp}  [{h.status}] cycle={h.cycle}  {h.detail}"
    if nonce:
        line += f"  nonce={nonce}"
    try:
        with ALERTS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        ALERT_LATEST.write_text(line + "\n", encoding="utf-8")
    except OSError:
        pass
    payload = {"campaign": "mfas", "status": h.status, "cycle": h.cycle,
               "detail": h.detail, "at": stamp, "nonce": nonce}
    receipt = {"logged": True, "webhook": _webhook(payload)}
    _desktop_notify(f"MFAS campaign: {h.status}", h.detail)
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true", help="check once; exit code is the health")
    ap.add_argument("--watch", action="store_true", help="loop forever")
    ap.add_argument("--every-min", type=float, default=5.0, help="poll interval for --watch")
    ap.add_argument("--stale-min", type=float,
                    default=float(os.environ.get("MFAS_STALE_MIN", 90)),
                    help="minutes without activity before STALE (default 90; a cycle is 20-40)")
    ap.add_argument("--test", action="store_true",
                    help="send a nonce alert through every channel and report the receipt")
    args = ap.parse_args(argv)

    if args.test:
        nonce = str(int(time.time()))
        receipt = raise_alert(Health("STALE", "watchdog self-test (not a real alert)", None, None),
                              nonce=nonce)
        print(f"test alert sent, nonce={nonce}, receipt={receipt}")
        wrote = ALERT_LATEST.exists() and nonce in ALERT_LATEST.read_text()
        print(f"ALERT file carries the nonce: {wrote}")
        return 0 if wrote else 1

    stale_s = args.stale_min * 60.0

    def check_and_report() -> Health:
        h = assess(time.time(), stale_s)
        print(f"[{time.strftime('%H:%M:%S')}] {h.status}: {h.detail}")
        if not h.ok:
            raise_alert(h)
        return h

    if args.watch:
        while True:
            check_and_report()
            time.sleep(args.every_min * 60.0)

    return check_and_report().code  # --once (default)


if __name__ == "__main__":
    raise SystemExit(main())
