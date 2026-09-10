#!/usr/bin/env python3
"""Smoke: sticky cancel flag must not cascade across subtasks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

from worker_agent import WorkerAgent  # noqa: E402


def main() -> None:
    w = WorkerAgent("http://127.0.0.1:5357", "smoke-worker", "smoke-host", sys.executable, Path("x.py"))
    w._mark_cancelled("manager")
    assert w._cancel_flag.is_set()
    assert w._cancel_reason == "manager"
    log_line, err = w._cancel_log_and_error()
    assert "manager" in log_line.lower()
    assert "manager" in err.lower()

    # Simulate end of cancelled subtask + start of next LAS
    w._reset_cancel_state()
    assert not w._cancel_flag.is_set(), "cancel flag must clear between subtasks"
    assert w._cancel_reason is None

    w._mark_cancelled("worker_stop")
    log_line, err = w._cancel_log_and_error()
    assert "worker stopping" in log_line.lower()
    assert "worker stopping" in err.lower()

    w._reset_cancel_state()
    assert not w._cancel_flag.is_set()
    print("ok: sticky cancel cascade guard")


if __name__ == "__main__":
    main()
