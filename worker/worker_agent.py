#!/usr/bin/env python3
"""TLS Classify cluster GPU worker.

Polls the manager for the next LAS subtask (one file at a time) and runs
classify_tls_gpu_0_0_1.py. Respects task cancellation and skips existing outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from las_result import code_identity, validate_result

POLL_IDLE = float(os.getenv("WORKER_POLL_IDLE_SEC", "5"))
POLL_BUSY = float(os.getenv("WORKER_HEARTBEAT_SEC", "10"))
LOG_FLUSH_CHARS = int(os.getenv("WORKER_LOG_FLUSH_CHARS", "4000"))
IO_RETRY_ATTEMPTS = int(os.getenv("WORKER_IO_RETRY_ATTEMPTS", "3"))
IO_RETRY_DELAY_SEC = float(os.getenv("WORKER_IO_RETRY_DELAY_SEC", "8"))
GPU_PREFLIGHT_RETRIES = int(os.getenv("WORKER_GPU_PREFLIGHT_RETRIES", "3"))
GPU_PREFLIGHT_DELAY_SEC = float(os.getenv("WORKER_GPU_PREFLIGHT_DELAY_SEC", "20"))


class WorkerAgent:
    def __init__(self, manager_url: str, worker_id: str, hostname: str, python_bin: str, script: Path):
        self.manager_url = manager_url.rstrip("/")
        self.worker_id = worker_id
        self.hostname = hostname
        self.python_bin = python_bin
        self.script = script
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._cancel_flag = threading.Event()
        # "manager" | "worker_stop" | None — why the current subtask was cancelled
        self._cancel_reason: str | None = None
        self._watch_stop = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._current_output: Path | None = None
        self._current_input: Path | None = None
        self._validation_subtask_id: int | None = None
        self._validation_heartbeat_at = 0.0

    def _reset_cancel_state(self) -> None:
        """Clear sticky cancel from a previous subtask (prevents cascade cancels)."""
        self._cancel_flag.clear()
        self._cancel_reason = None

    def _mark_cancelled(self, reason: str) -> None:
        """reason: 'manager' (task cancelled) or 'worker_stop' (SIGTERM / agent stop)."""
        self._cancel_reason = reason
        self._cancel_flag.set()

    def _cancel_log_and_error(self) -> tuple[str, str]:
        reason = self._cancel_reason
        if reason is None and self._stop.is_set():
            reason = "worker_stop"
        if reason == "worker_stop":
            return "Cancelled: worker stopping\n", "Cancelled: worker stopping"
        return "Cancelled by manager\n", "Cancelled by manager"

    def _url(self, path: str) -> str:
        return f"{self.manager_url}{path}"

    @staticmethod
    def _cleanup_writing(output_file: Path | None) -> None:
        if output_file is None:
            return
        # Match write_las_atomic temp names: name.las.<pid>.writing and legacy name.las.writing
        parent = output_file.parent
        stem = output_file.name
        patterns = [f"{stem}.writing", f"{stem}.*.writing"]
        for pat in patterns:
            for p in parent.glob(pat) if "*" in pat else [parent / pat]:
                try:
                    if p.is_file():
                        p.unlink()
                        print(f"[worker] removed orphan temp {p}", flush=True)
                except OSError as e:
                    print(f"[worker] temp cleanup failed {p}: {e}", flush=True)

    @staticmethod
    def _is_retryable_io_error(text: str) -> bool:
        t = text.lower()
        needles = (
            "filenotfounderror",
            "permissionerror",
            "stale file handle",
            "input/output error",
            "network is unreachable",
            "network name cannot be found",
            "timed out",
            "resource temporarily unavailable",
            "device or resource busy",
            ".writing",
            "errno 2",
            "errno 5",
            "errno 13",
            "errno 16",
            "errno 110",
        )
        return any(n in t for n in needles)

    @staticmethod
    def _is_gpu_missing_error(text: str) -> bool:
        t = text.lower()
        return "cudaerrornodevice" in t or "no cuda-capable device is detected" in t

    def _validation_checkpoint(self) -> None:
        if self._stop.is_set():
            self._mark_cancelled('worker_stop')
        if self._cancel_flag.is_set():
            raise InterruptedError('LAS verification cancelled')
        if self._validation_heartbeat_at == 0.0 or time.monotonic() - self._validation_heartbeat_at >= POLL_BUSY:
            self._validation_heartbeat_at = time.monotonic()
            try:
                if self.heartbeat(self._validation_subtask_id):
                    self._mark_cancelled('manager')
                    raise InterruptedError('LAS verification cancelled by manager')
            except requests.RequestException as exc:
                print(f'[worker] validation heartbeat failed: {exc}', flush=True)

    def _check_output_exists_with_retry(self, output_file: Path, provenance=None) -> bool:
        for attempt in range(1, IO_RETRY_ATTEMPTS + 1):
            try:
                if self._current_input is None:
                    raise RuntimeError('Cannot validate output without the input path')
                return validate_result(self._current_input, output_file, provenance,
                                       checkpoint=self._validation_checkpoint)
            except InterruptedError:
                raise
            except OSError as e:
                if attempt >= IO_RETRY_ATTEMPTS:
                    print(f"[worker] output stat failed after retries: {e}", flush=True)
                    raise OSError(f'OUTPUT_CHECK_IO_FAILED: {output_file}') from e
                print(
                    f"[worker] output stat failed ({attempt}/{IO_RETRY_ATTEMPTS}), "
                    f"retry in {IO_RETRY_DELAY_SEC:.1f}s: {e}",
                    flush=True,
                )
                time.sleep(IO_RETRY_DELAY_SEC)
        return False

    def _ensure_output_dir_with_retry(self, output_file: Path) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, IO_RETRY_ATTEMPTS + 1):
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                return
            except OSError as e:
                last_exc = e
                if attempt >= IO_RETRY_ATTEMPTS:
                    break
                print(
                    f"[worker] mkdir failed ({attempt}/{IO_RETRY_ATTEMPTS}), "
                    f"retry in {IO_RETRY_DELAY_SEC:.1f}s: {e}",
                    flush=True,
                )
                time.sleep(IO_RETRY_DELAY_SEC)
        raise OSError(f"Cannot create output directory after retries: {output_file.parent}") from last_exc

    def _cleanup_stale_classify_processes(self) -> None:
        if os.name == "nt":
            return
        script_name = self.script.name
        try:
            proc = subprocess.run(
                ["ps", "-eo", "pid=,args="],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as e:
            print(f"[worker] stale process scan skipped: {e}", flush=True)
            return
        my_pid = os.getpid()
        stale_pids: list[int] = []
        for line in proc.stdout.splitlines():
            m = re.match(r"^\s*(\d+)\s+(.*)$", line)
            if not m:
                continue
            pid = int(m.group(1))
            args = m.group(2)
            if pid == my_pid:
                continue
            if script_name in args and "worker_agent.py" not in args:
                stale_pids.append(pid)
        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"[worker] terminated stale classify process pid={pid}", flush=True)
            except ProcessLookupError:
                continue
            except Exception as e:
                print(f"[worker] stale classify terminate failed pid={pid}: {e}", flush=True)

    def _start_cancel_watch(self, subtask_id: int) -> None:
        # Cancel state is reset only at the start of run_classify (_reset_cancel_state).
        # Clearing here would drop a manager-cancel detected before Popen / between attempts.
        self._watch_stop.clear()

        def _watch() -> None:
            while not self._watch_stop.wait(POLL_BUSY):
                if self._stop.is_set():
                    self._mark_cancelled("worker_stop")
                    self._terminate_proc()
                    return
                try:
                    if self.heartbeat(subtask_id):
                        print("[worker] cancel requested (watch thread)", flush=True)
                        self._mark_cancelled("manager")
                        self._terminate_proc()
                        return
                except Exception as e:
                    print(f"[worker] cancel-watch heartbeat failed: {e}", flush=True)

        self._watch_thread = threading.Thread(target=_watch, name="cancel-watch", daemon=True)
        self._watch_thread.start()

    def _stop_cancel_watch(self) -> None:
        self._watch_stop.set()
        t = self._watch_thread
        self._watch_thread = None
        if t and t.is_alive():
            t.join(timeout=5)

    def register(self) -> None:
        r = requests.post(
            self._url("/api/workers/register"),
            json={"worker_id": self.worker_id, "hostname": self.hostname},
            timeout=30,
        )
        r.raise_for_status()
        print(f"[worker] registered as {self.worker_id} @ {self.hostname}", flush=True)

    def claim(self) -> dict | None:
        r = requests.post(
            self._url("/api/workers/claim"),
            json={"worker_id": self.worker_id, "hostname": self.hostname},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("subtask")

    def heartbeat(self, subtask_id: int | None) -> bool:
        r = requests.post(
            self._url("/api/workers/heartbeat"),
            json={"worker_id": self.worker_id, "current_subtask_id": subtask_id},
            timeout=30,
        )
        r.raise_for_status()
        return bool(r.json().get("cancel_requested"))

    def progress(self, subtask_id: int, progress: float, log_append: str = "") -> bool:
        """Return True if parent task was cancelled (via cancel-check after progress)."""
        r = requests.post(
            self._url(f"/api/workers/subtasks/{subtask_id}/progress"),
            json={
                "worker_id": self.worker_id,
                "progress": progress,
                "log_append": log_append,
            },
            timeout=60,
        )
        r.raise_for_status()
        c = requests.get(
            self._url(f"/api/workers/subtasks/{subtask_id}/cancel-check"),
            timeout=30,
        )
        c.raise_for_status()
        return bool(c.json().get("cancelled"))

    def complete(
        self,
        subtask_id: int,
        status: str,
        log_append: str = "",
        error_message: str | None = None,
        progress: float | None = None,
    ) -> None:
        r = requests.post(
            self._url(f"/api/workers/subtasks/{subtask_id}/complete"),
            json={
                "worker_id": self.worker_id,
                "status": status,
                "log_append": log_append,
                "error_message": error_message,
                "progress": progress,
            },
            timeout=60,
        )
        r.raise_for_status()

    def _terminate_proc(self) -> None:
        proc = self._proc
        if not proc or proc.poll() is not None:
            return
        print("[worker] terminating classify process (cancel)", flush=True)
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=30)
        except Exception:
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        self._cleanup_writing(self._current_output)

    def run_classify(self, subtask: dict) -> tuple[str, str]:
        """Returns (status, error_message)."""
        # Critical: never inherit cancel flag from a previous LAS on this worker.
        self._reset_cancel_state()

        input_file = Path(subtask["input_file"])
        self._current_input = input_file
        output_file = Path(subtask["output_file"])
        subtask_id = subtask["id"]
        self._validation_subtask_id = subtask_id
        self._validation_heartbeat_at = 0.0
        self._current_output = output_file

        if self._check_output_exists_with_retry(output_file):
            self.complete(
                subtask_id,
                "skipped",
                log_append="Skipped by worker: complete LAS verified against input\n",
                progress=100.0,
            )
            self._current_output = None
            return "skipped", None

        self._ensure_output_dir_with_retry(output_file)
        self._cleanup_stale_classify_processes()

        if os.getenv("WORKER_MODE", "gpu").lower() == "mock":
            try:
                return self._run_mock(subtask_id, input_file, output_file)
            finally:
                self._current_output = None
                self._reset_cancel_state()

        cmd = [
            self.python_bin,
            str(self.script),
            str(input_file),
            "-o",
            str(output_file),
            "--cars-to",
            os.getenv("CARS_TO", "91"),
            "--noise-to",
            os.getenv("NOISE_TO", "7"),
        ]
        env = os.environ.copy()
        pipeline_root = str(self.script.resolve().parent.parent)
        env["PYTHONPATH"] = pipeline_root + os.pathsep + env.get("PYTHONPATH", "")
        progress_val = 1.0

        try:
            if self.progress(subtask_id, progress_val, log_append=f"$ {' '.join(cmd)}\n"):
                self._mark_cancelled("manager")
                log_line, err_msg = self._cancel_log_and_error()
                self._cleanup_writing(output_file)
                self.complete(
                    subtask_id,
                    "cancelled",
                    log_append=log_line,
                    error_message=err_msg,
                    progress=progress_val,
                )
                self._current_output = None
                self._reset_cancel_state()
                return "cancelled", err_msg
        except Exception as e:
            print(f"[worker] initial progress failed: {e}", flush=True)

        if self._stop.is_set():
            self._mark_cancelled("worker_stop")
            log_line, err_msg = self._cancel_log_and_error()
            self._cleanup_writing(output_file)
            self.complete(
                subtask_id,
                "cancelled",
                log_append=log_line,
                error_message=err_msg,
                progress=progress_val,
            )
            self._current_output = None
            self._reset_cancel_state()
            return "cancelled", err_msg

        rc = -1
        tail = ""
        cancelled = False
        io_retry_note = ""
        for attempt in range(1, IO_RETRY_ATTEMPTS + 1):
            if self._stop.is_set():
                self._mark_cancelled("worker_stop")
                cancelled = True
                break
            if self._cancel_flag.is_set():
                cancelled = True
                break
            if attempt > 1:
                note = (
                    f"[worker] retry classify attempt {attempt}/{IO_RETRY_ATTEMPTS} "
                    f"after transient I/O or GPU preflight issue\n"
                )
                io_retry_note += note
                try:
                    if self.progress(subtask_id, progress_val, log_append=note):
                        self._mark_cancelled("manager")
                        cancelled = True
                        break
                except Exception as e:
                    print(f"[worker] progress upload failed: {e}", flush=True)
            print(f"[worker] starting attempt {attempt}/{IO_RETRY_ATTEMPTS}: {' '.join(cmd)}", flush=True)
            preexec = None if os.name == "nt" else os.setsid
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                preexec_fn=preexec,
            )
            self._start_cancel_watch(subtask_id)

            buf: list[str] = []
            chars = 0
            last_flush = time.time()
            try:
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    if self._cancel_flag.is_set() or self._stop.is_set():
                        if self._stop.is_set() and self._cancel_reason != "manager":
                            self._mark_cancelled("worker_stop")
                        cancelled = True
                        break
                    buf.append(line)
                    chars += len(line)
                    progress_val = min(95.0, progress_val + 0.05)
                    now = time.time()
                    if chars >= LOG_FLUSH_CHARS or (now - last_flush) >= POLL_BUSY:
                        chunk = "".join(buf)
                        buf.clear()
                        chars = 0
                        last_flush = now
                        try:
                            if self.progress(subtask_id, progress_val, log_append=chunk):
                                self._mark_cancelled("manager")
                                cancelled = True
                                self._terminate_proc()
                                break
                        except Exception as e:
                            print(f"[worker] progress upload failed: {e}", flush=True)
                if self._cancel_flag.is_set() or self._stop.is_set():
                    if self._stop.is_set() and self._cancel_reason != "manager":
                        self._mark_cancelled("worker_stop")
                    cancelled = True
                    self._terminate_proc()
                rc = self._proc.wait() if self._proc else -1
            finally:
                self._stop_cancel_watch()
                self._proc = None
            tail = "".join(buf)
            if cancelled:
                break
            if rc == 0 and self._check_output_exists_with_retry(output_file):
                break
            retryable = self._is_retryable_io_error(tail)
            gpu_missing = self._is_gpu_missing_error(tail)
            if gpu_missing and attempt < min(IO_RETRY_ATTEMPTS, GPU_PREFLIGHT_RETRIES):
                wait_s = max(IO_RETRY_DELAY_SEC, GPU_PREFLIGHT_DELAY_SEC)
                note = (
                    f"[worker] GPU preflight failed (no device), retry in {wait_s:.1f}s "
                    f"({attempt}/{GPU_PREFLIGHT_RETRIES})\n"
                )
                print(note.strip(), flush=True)
                try:
                    if self.progress(subtask_id, progress_val, log_append=note):
                        self._mark_cancelled("manager")
                        cancelled = True
                        break
                except Exception as e:
                    print(f"[worker] progress upload failed: {e}", flush=True)
                time.sleep(wait_s)
                continue
            if retryable and attempt < IO_RETRY_ATTEMPTS:
                note = (
                    f"[worker] transient storage I/O issue, retry in {IO_RETRY_DELAY_SEC:.1f}s "
                    f"({attempt}/{IO_RETRY_ATTEMPTS})\n"
                )
                print(note.strip(), flush=True)
                try:
                    if self.progress(subtask_id, progress_val, log_append=note):
                        self._mark_cancelled("manager")
                        cancelled = True
                        break
                except Exception as e:
                    print(f"[worker] progress upload failed: {e}", flush=True)
                time.sleep(IO_RETRY_DELAY_SEC)
                continue
            break

        if cancelled:
            self._cleanup_writing(output_file)
            log_line, err_msg = self._cancel_log_and_error()
            self.complete(
                subtask_id,
                "cancelled",
                log_append=tail + "\n" + log_line,
                error_message=err_msg,
                progress=progress_val,
            )
            self._current_output = None
            self._reset_cancel_state()
            return "cancelled", err_msg

        if rc == 0 and self._check_output_exists_with_retry(output_file):
            self._check_output_exists_with_retry(output_file, provenance=code_identity(self.script))
            self.complete(
                subtask_id,
                "success",
                log_append=io_retry_note + tail + f"\nExit code {rc}\n",
                progress=100.0,
            )
            self._current_output = None
            self._reset_cancel_state()
            return "success", None

        err = f"Classify failed with exit code {rc}"
        if not self._check_output_exists_with_retry(output_file):
            err += " (no output file)"
        self._cleanup_writing(output_file)
        self.complete(
            subtask_id,
            "error",
            log_append=io_retry_note + tail + f"\n{err}\n",
            error_message=err,
            progress=progress_val,
        )
        self._current_output = None
        self._reset_cancel_state()
        return "error", err

    def _run_mock(self, subtask_id: int, input_file: Path, output_file: Path) -> tuple[str, str]:
        """Smoke-test mode: copy input to output with progress heartbeats."""
        import shutil

        self._reset_cancel_state()
        if self.progress(subtask_id, 10.0, log_append=f"[mock] copy {input_file} -> {output_file}\n"):
            self._mark_cancelled("manager")
            log_line, err_msg = self._cancel_log_and_error()
            self.complete(subtask_id, "cancelled", log_append=log_line, error_message=err_msg, progress=10.0)
            return "cancelled", err_msg
        for step, pct in enumerate((30.0, 60.0, 90.0), start=1):
            if self._stop.is_set():
                self._mark_cancelled("worker_stop")
                log_line, err_msg = self._cancel_log_and_error()
                self.complete(subtask_id, "cancelled", log_append=log_line, error_message=err_msg, progress=pct)
                return "cancelled", err_msg
            if self.heartbeat(subtask_id):
                self._mark_cancelled("manager")
                log_line, err_msg = self._cancel_log_and_error()
                self.complete(subtask_id, "cancelled", log_append=log_line, error_message=err_msg, progress=pct)
                return "cancelled", err_msg
            time.sleep(float(os.getenv("MOCK_STEP_SEC", "1")))
            if self.progress(subtask_id, pct, log_append=f"[mock] step {step}\n"):
                self._mark_cancelled("manager")
                log_line, err_msg = self._cancel_log_and_error()
                self.complete(subtask_id, "cancelled", log_append=log_line, error_message=err_msg, progress=pct)
                return "cancelled", err_msg
        shutil.copy2(input_file, output_file)
        self.complete(subtask_id, "success", log_append="[mock] done\n", progress=100.0)
        return "success", None

    def loop(self) -> None:
        self.register()
        while not self._stop.is_set():
            try:
                sub = self.claim()
            except Exception as e:
                print(f"[worker] claim error: {e}", flush=True)
                time.sleep(POLL_IDLE)
                continue

            if not sub:
                time.sleep(POLL_IDLE)
                continue

            print(
                f"[worker] claimed subtask #{sub['id']} {sub.get('filename')} "
                f"-> {sub.get('output_file')}",
                flush=True,
            )
            try:
                status, err = self.run_classify(sub)
                print(f"[worker] finished #{sub['id']}: {status} {err or ''}", flush=True)
            except Exception as e:
                print(f"[worker] run error: {e}", flush=True)
                try:
                    self.complete(
                        sub["id"],
                        "cancelled" if self._cancel_flag.is_set() else "error",
                        log_append=f"\nWorker exception: {e}\n",
                        error_message=str(e),
                    )
                except Exception as e2:
                    print(f"[worker] complete failed: {e2}", flush=True)

    def stop(self, *_args) -> None:
        self._stop.set()
        self._terminate_proc()


def default_script() -> Path:
    env = os.getenv("CLASSIFY_SCRIPT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "pipeline" / "GPU_0_0_1" / "classify_tls_gpu_0_0_1.py",
        Path("/opt/tls/pipeline/GPU_0_0_1/classify_tls_gpu_0_0_1.py"),
        here / "classify_tls_gpu_0_0_1.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser(description="TLS Classify cluster worker")
    p.add_argument("--manager-url", default=os.getenv("MANAGER_URL", "http://127.0.0.1:5357"))
    p.add_argument(
        "--worker-id",
        default=os.getenv("WORKER_ID") or socket.gethostname(),
    )
    p.add_argument("--hostname", default=os.getenv("HOSTNAME") or socket.gethostname())
    p.add_argument("--python", default=os.getenv("PYTHON_BIN", sys.executable))
    p.add_argument("--script", type=Path, default=default_script())
    args = p.parse_args()

    agent = WorkerAgent(args.manager_url, args.worker_id, args.hostname, args.python, args.script)
    signal.signal(signal.SIGINT, agent.stop)
    signal.signal(signal.SIGTERM, agent.stop)
    print(f"[worker] manager={args.manager_url} script={args.script}", flush=True)
    agent.loop()


if __name__ == "__main__":
    main()
