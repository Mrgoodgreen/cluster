from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, aliased

from .config import settings
from .models import Subtask, Task, Worker, utcnow

LAS_EXTS = {".las", ".laz"}
TERMINAL_TASK = {"completed", "error", "cancelled"}
TERMINAL_SUB = {"skipped", "success", "error", "cancelled"}
ACTIVE_TASK = {"pending", "processing"}


def _trim_log(text: str) -> str:
    max_chars = settings.log_text_max_chars
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _as_utc(dt: datetime | None) -> datetime | None:
    """Normalize DB datetimes: SQLite often returns naive UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def display_to_local(display_path: str) -> Path:
    """Map UI path like '172.16.16.126/foo/bar' to local filesystem path."""
    display_path = display_path.strip().replace("\\", "/").strip("/")
    if not display_path:
        raise HTTPException(400, "Empty path")

    parts = display_path.split("/")
    root_key = parts[0]
    roots = settings.storage_roots
    if root_key not in roots:
        raise HTTPException(400, f"Unknown storage root: {root_key}")
    local = roots[root_key]
    if len(parts) > 1:
        local = local.joinpath(*parts[1:])
    resolved = local.resolve()
    root_resolved = roots[root_key].resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise HTTPException(400, "Path escapes storage root") from exc
    return resolved


def local_to_display(local: Path) -> str | None:
    local = local.resolve()
    for key, root in settings.storage_roots.items():
        root_r = root.resolve()
        try:
            rel = local.relative_to(root_r)
        except ValueError:
            continue
        rel_s = str(rel).replace("\\", "/")
        return key if rel_s in ("", ".") else f"{key}/{rel_s}"
    return None


def mkdir_storage(display_path: str) -> dict:
    """Create a folder under STORAGE_ROOTS (parents=True)."""
    display = display_path.strip().replace("\\", "/").strip("/")
    if not display:
        raise HTTPException(400, "Empty path")
    parts = display.split("/")
    if len(parts) < 2:
        raise HTTPException(400, "Specify a folder name under a storage root (root/name)")
    local = display_to_local(display)
    root_key = parts[0]
    root = settings.storage_roots[root_key].resolve()
    if local.resolve() == root:
        raise HTTPException(400, "Cannot create storage root")
    existed = local.exists()
    if existed and not local.is_dir():
        raise HTTPException(400, f"Path exists and is not a directory: {display}")
    if not existed:
        try:
            local.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(500, f"Cannot create directory: {exc}") from exc
    disp = local_to_display(local) or display
    return {"path": disp, "created": not existed, "existed": existed}


def list_storage_children(display_path: str | None) -> list[dict]:
    if not display_path or display_path.strip() in ("", "/"):
        nodes = []
        for key, root in sorted(settings.storage_roots.items()):
            exists = root.exists()
            nodes.append(
                {
                    "title": key,
                    "key": key,
                    "path": key,
                    "is_leaf": False,
                    "disabled": not exists,
                }
            )
        return nodes

    local = display_to_local(display_path)
    if not local.exists() or not local.is_dir():
        return []

    children: list[dict] = []
    try:
        entries = sorted(local.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise HTTPException(403, f"Cannot read directory: {exc}") from exc

    for entry in entries:
        if entry.name.startswith("."):
            continue
        # tree shows directories only (files selected via folder path)
        if not entry.is_dir():
            continue
        disp = local_to_display(entry) or f"{display_path.rstrip('/')}/{entry.name}"
        children.append(
            {
                "title": entry.name,
                "key": disp,
                "path": disp,
                "is_leaf": False,
            }
        )
    return children


def _output_for(input_file: Path, input_root: Path, output_root: Path) -> Path:
    try:
        rel = input_file.relative_to(input_root)
    except ValueError:
        rel = Path(input_file.name)
    return output_root / rel


def scan_las_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(400, f"Input path is not a directory: {folder}")
    files = [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in LAS_EXTS
    ]
    return files


def create_task(db: Session, input_display: str, output_display: str) -> Task:
    input_local = display_to_local(input_display)
    output_local = display_to_local(output_display)
    if not input_local.exists() or not input_local.is_dir():
        raise HTTPException(400, f"Input directory not found: {input_display}")
    output_local.mkdir(parents=True, exist_ok=True)

    files = scan_las_files(input_local)
    if not files:
        raise HTTPException(400, "No .las/.laz files found under input path")

    task = Task(
        uid=str(uuid.uuid4()),
        status="pending",
        input_path=input_display.strip().replace("\\", "/").strip("/"),
        output_path=output_display.strip().replace("\\", "/").strip("/"),
    )
    db.add(task)
    db.flush()

    for f in files:
        out = _output_for(f, input_local, output_local)
        rel = str(f.relative_to(input_local)).replace("\\", "/")
        # Verification belongs to the worker: a non-empty file can be truncated.
        # Keep storage I/O out of the manager's task creation transaction.
        st = Subtask(
            task_id=task.id,
            filename=f.name,
            relative_path=rel,
            input_file=str(f.resolve()),
            output_file=str(out.resolve()),
            status="pending",
            progress=0.0,
            finished_at=None,
            log_text="",
        )
        db.add(st)

    db.flush()
    _maybe_finalize_task(db, task)
    db.commit()
    db.refresh(task)
    return task


def restart_task(db: Session, task_id: int) -> Task:
    src = db.get(Task, task_id)
    if not src:
        raise HTTPException(404, "Task not found")
    return create_task(db, src.input_path, src.output_path)


def task_counts(db: Session, task_id: int) -> tuple[int, int]:
    total = db.scalar(select(func.count()).select_from(Subtask).where(Subtask.task_id == task_id)) or 0
    done = (
        db.scalar(
            select(func.count())
            .select_from(Subtask)
            .where(Subtask.task_id == task_id, Subtask.status.in_(tuple(TERMINAL_SUB)))
        )
        or 0
    )
    return int(total), int(done)


def serialize_task(db: Session, task: Task, with_subtasks: bool = False) -> dict:
    total, done = task_counts(db, task.id)
    data = {
        "id": task.id,
        "uid": task.uid,
        "status": task.status,
        "input_path": task.input_path,
        "output_path": task.output_path,
        "created_at": task.created_at,
        "finished_at": task.finished_at,
        "error_message": task.error_message,
        "subtask_total": total,
        "subtask_done": done,
    }
    if with_subtasks:
        subs = db.scalars(
            select(Subtask).where(Subtask.task_id == task.id).order_by(Subtask.id)
        ).all()
        # UI gets display SMB paths; workers still receive absolute paths via /claim
        data["subtasks"] = [
            {
                "id": s.id,
                "task_id": s.task_id,
                "filename": s.filename,
                "relative_path": s.relative_path,
                "input_file": local_to_display(Path(s.input_file)) or s.input_file,
                "output_file": local_to_display(Path(s.output_file)) or s.output_file,
                "status": s.status,
                "worker_id": s.worker_id,
                "started_at": s.started_at,
                "finished_at": s.finished_at,
                "progress": s.progress,
                "log_text": s.log_text or "",
                "error_message": s.error_message,
            }
            for s in subs
        ]
    return data


def _maybe_finalize_task(db: Session, task: Task) -> None:
    if task.status == "cancelled":
        return
    subs = db.scalars(select(Subtask).where(Subtask.task_id == task.id)).all()
    if not subs:
        task.status = "completed"
        task.finished_at = utcnow()
        return
    if any(s.status == "pending" or s.status == "processing" for s in subs):
        if task.status == "pending" and any(s.status == "processing" for s in subs):
            task.status = "processing"
        return
    # all terminal
    if any(s.status == "cancelled" for s in subs) and all(
        s.status in TERMINAL_SUB for s in subs
    ):
        # if cancel drained remaining
        if all(s.status == "cancelled" or s.status == "skipped" or s.status == "success" for s in subs):
            if any(s.status == "cancelled" for s in subs):
                task.status = "cancelled"
            else:
                task.status = "completed"
        elif any(s.status == "error" for s in subs):
            task.status = "error"
            task.error_message = "One or more subtasks failed"
        else:
            task.status = "cancelled"
    elif any(s.status == "error" for s in subs):
        task.status = "error"
        task.error_message = "One or more subtasks failed"
    else:
        task.status = "completed"
    task.finished_at = utcnow()


def cancel_task(db: Session, task_id: int) -> Task:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status in TERMINAL_TASK:
        return task

    task.status = "cancelled"
    task.finished_at = utcnow()
    for st in db.scalars(select(Subtask).where(Subtask.task_id == task.id)).all():
        if st.status == "pending":
            st.status = "cancelled"
            st.finished_at = utcnow()
            st.log_text = (st.log_text or "") + "Cancelled before start\n"
        # processing left as-is; worker will observe cancel and complete as cancelled
    db.commit()
    db.refresh(task)
    return task


def register_worker(db: Session, worker_id: str, hostname: str, *, commit: bool = True) -> Worker:
    w = db.get(Worker, worker_id)
    if w is None:
        w = Worker(id=worker_id, hostname=hostname or worker_id)
        db.add(w)
    else:
        w.hostname = hostname or w.hostname
        w.last_heartbeat = utcnow()
    if commit:
        db.commit()
        db.refresh(w)
    return w


def heartbeat(db: Session, worker_id: str, current_subtask_id: int | None) -> Worker:
    w = db.get(Worker, worker_id)
    if w is None:
        w = Worker(id=worker_id, hostname=worker_id)
        db.add(w)
    w.last_heartbeat = utcnow()
    w.current_subtask_id = current_subtask_id
    db.commit()
    db.refresh(w)
    return w


def reclaim_stale_processing(db: Session) -> int:
    """Return stuck processing subtasks to pending when worker lease expires."""
    cutoff = utcnow() - timedelta(minutes=settings.worker_lease_minutes)
    n = 0
    processing = db.scalars(select(Subtask).where(Subtask.status == "processing")).all()
    for st in processing:
        w = db.get(Worker, st.worker_id) if st.worker_id else None
        hb = _as_utc(w.last_heartbeat) if w else None
        started = _as_utc(st.started_at)
        stale = w is None or hb is None or hb < cutoff
        if (
            not stale
            and started is not None
            and started < cutoff
            and (w is None or w.current_subtask_id != st.id)
        ):
            stale = True
        if not stale:
            continue
        st.status = "pending"
        st.worker_id = None
        st.started_at = None
        st.progress = 0.0
        st.log_text = _trim_log(
            (st.log_text or "")
            + f"Reclaimed: worker lease expired ({settings.worker_lease_minutes} min)\n"
        )
        if w and w.current_subtask_id == st.id:
            w.current_subtask_id = None
        n += 1
    return n


def claim_subtask(db: Session, worker_id: str, hostname: str) -> Subtask | None:
    """Claim next pending LAS in global FIFO with atomic UPDATE … WHERE pending."""
    register_worker(db, worker_id, hostname, commit=False)
    reclaim_stale_processing(db)

    active = db.scalars(
        select(Task).where(Task.status.in_(tuple(ACTIVE_TASK))).order_by(Task.id.asc())
    ).all()
    for task in active:
        _maybe_finalize_task(db, task)
    db.commit()

    candidates = db.execute(
        select(Subtask.id, Subtask.output_file, Subtask.task_id)
        .join(Task, Subtask.task_id == Task.id)
        .where(
            Subtask.status == "pending",
            Task.status.in_(tuple(ACTIVE_TASK)),
        )
        .order_by(Task.id.asc(), Subtask.id.asc())
    ).all()

    for sid, output_file, task_id in candidates:
        task = db.get(Task, task_id)
        if task is None or task.status not in ACTIVE_TASK:
            continue

        # The worker verifies records/coordinates (or a matching completion receipt)
        # before reporting skipped. Never skip here based only on stat().

        other_output = aliased(Subtask)
        output_busy = select(other_output.id).where(
            other_output.output_file == output_file,
            other_output.status == 'processing',
        ).exists()
        result = db.execute(
            update(Subtask)
            .where(Subtask.id == sid, Subtask.status == "pending", ~output_busy)
            .values(
                status="processing",
                worker_id=worker_id,
                started_at=utcnow(),
                progress=0.0,
                finished_at=None,
            )
        )
        if result.rowcount != 1:
            db.rollback()
            continue

        db.refresh(task)
        if task.status == "cancelled":
            db.execute(
                update(Subtask)
                .where(Subtask.id == sid)
                .values(
                    status="cancelled",
                    finished_at=utcnow(),
                    worker_id=None,
                    log_text="Cancelled while claiming\n",
                )
            )
            db.commit()
            continue

        if task.status == "pending":
            task.status = "processing"
        w = db.get(Worker, worker_id)
        if w:
            w.current_subtask_id = sid
            w.last_heartbeat = utcnow()
        db.commit()
        st = db.get(Subtask, sid)
        return st

    return None


def is_task_cancelled(db: Session, task_id: int) -> bool:
    task = db.get(Task, task_id)
    return bool(task and task.status == "cancelled")


def update_progress(db: Session, subtask_id: int, worker_id: str, progress: float, log_append: str) -> Subtask:
    st = db.get(Subtask, subtask_id)
    if not st:
        raise HTTPException(404, "Subtask not found")
    if st.worker_id and st.worker_id != worker_id:
        raise HTTPException(409, "Subtask owned by another worker")
    if st.status not in {"processing", "pending"}:
        raise HTTPException(409, f"Subtask not updatable in status={st.status}")
    st.progress = max(0.0, min(100.0, float(progress)))
    if log_append:
        st.log_text = _trim_log((st.log_text or "") + log_append)
    w = db.get(Worker, worker_id)
    if w:
        w.last_heartbeat = utcnow()
        w.current_subtask_id = st.id
    db.commit()
    db.refresh(st)
    return st


def complete_subtask(
    db: Session,
    subtask_id: int,
    worker_id: str,
    status: str,
    log_append: str,
    error_message: str | None,
    progress: float | None,
) -> Subtask:
    if status not in {"success", "error", "skipped", "cancelled"}:
        raise HTTPException(400, "Invalid completion status")
    st = db.get(Subtask, subtask_id)
    if not st:
        raise HTTPException(404, "Subtask not found")
    if st.worker_id and st.worker_id != worker_id:
        raise HTTPException(409, "Subtask owned by another worker")

    task = db.get(Task, st.task_id)
    if task and task.status == "cancelled" and status not in {"success", "skipped"}:
        status = "cancelled"

    st.status = status
    st.finished_at = utcnow()
    if progress is not None:
        st.progress = max(0.0, min(100.0, float(progress)))
    elif status in {"success", "skipped"}:
        st.progress = 100.0
    if log_append:
        st.log_text = _trim_log((st.log_text or "") + log_append)
    if error_message:
        st.error_message = error_message

    w = db.get(Worker, worker_id)
    if w:
        w.current_subtask_id = None
        w.last_heartbeat = utcnow()

    if task:
        _maybe_finalize_task(db, task)
    db.commit()
    db.refresh(st)
    return st
