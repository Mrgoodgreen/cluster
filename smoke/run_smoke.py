#!/usr/bin/env python3
"""Smoke-test the cluster API against docker-compose.smoke.yml."""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import json
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5357"
STORAGE = Path(__file__).resolve().parent / "storage"


def http(method: str, path: str, body: dict | None = None) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http("GET", "/api/health")
            print("health ok")
            return
        except Exception as e:
            print(f"waiting for manager: {e}")
            time.sleep(2)
    raise SystemExit("manager did not become healthy")


def wait_task(task_id: int, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = http("GET", f"/api/tasks/{task_id}")
        print(f"  task {task_id}: {t['status']} {t['subtask_done']}/{t['subtask_total']}")
        if t["status"] in {"completed", "error", "cancelled"}:
            return t
        time.sleep(2)
    raise SystemExit(f"task {task_id} timeout")


def main() -> None:
    wait_health()

    # 1) Process both files
    t1 = http(
        "POST",
        "/api/tasks",
        {"input_path": "smoke/in", "output_path": "smoke/out"},
    )
    print(f"created task {t1['id']}")
    t1 = wait_task(t1["id"])
    assert t1["status"] == "completed", t1
    assert (STORAGE / "out" / "a.las").exists()
    assert (STORAGE / "out" / "sub" / "b.las").exists()
    print("PASS: first task completed and outputs exist")

    # 2) Re-run — should skip
    t2 = http(
        "POST",
        "/api/tasks",
        {"input_path": "smoke/in", "output_path": "smoke/out"},
    )
    print(f"created task {t2['id']} (expect skips)")
    # May already be completed at create time if all skipped
    t2 = wait_task(t2["id"], timeout=60)
    assert t2["status"] == "completed", t2
    statuses = {s["status"] for s in t2["subtasks"]}
    assert statuses == {"skipped"}, statuses
    print("PASS: second task fully skipped")

    # 3) Cancel path — wipe out so work would be needed, then cancel quickly
    for p in (STORAGE / "out").rglob("*"):
        if p.is_file():
            p.unlink()
    t3 = http(
        "POST",
        "/api/tasks",
        {"input_path": "smoke/in", "output_path": "smoke/out"},
    )
    print(f"created task {t3['id']} then cancel")
    http("POST", f"/api/tasks/{t3['id']}/cancel")
    t3 = wait_task(t3["id"], timeout=60)
    assert t3["status"] == "cancelled", t3
    print("PASS: cancel")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        raise SystemExit(f"HTTP error: {e}") from e
