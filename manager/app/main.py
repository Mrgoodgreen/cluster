from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, or_
from sqlalchemy.orm import Session

from .config import settings
from .models import SessionLocal, Task, init_db
from . import schemas, services

app = FastAPI(title="TLS Classify Cluster", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# Ensure DB is ready even if lifespan hooks differ across uvicorn versions
init_db()

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "storage_roots": list(settings.storage_roots.keys()),
    }


@app.get("/api/storage/tree")
def storage_tree(path: str | None = Query(None)):
    return {"nodes": services.list_storage_children(path)}


@app.post("/api/storage/mkdir", response_model=schemas.StorageMkdirResponse)
def storage_mkdir(body: schemas.StorageMkdirRequest):
    return services.mkdir_storage(body.path)


@app.post("/api/tasks", response_model=schemas.TaskDetail)
def create_task(body: schemas.TaskCreate, db: Session = Depends(get_db)):
    task = services.create_task(db, body.input_path, body.output_path)
    return services.serialize_task(db, task, with_subtasks=True)


@app.get("/api/tasks", response_model=schemas.TaskList)
def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    status: Literal['pending', 'processing', 'completed', 'error', 'cancelled'] | None = None,
    q: str = Query('', max_length=200),
):
    filters = []
    if status:
        filters.append(Task.status == status)
    if q.strip():
        term = q.strip()
        matches = [Task.input_path.icontains(term, autoescape=True),
                   Task.output_path.icontains(term, autoescape=True),
                   Task.uid.icontains(term, autoescape=True)]
        if term.isascii() and term.isdigit() and len(term) < 19:
            matches.append(Task.id == int(term))
        filters.append(or_(*matches))
    total = db.scalar(select(func.count()).select_from(Task).where(*filters)) or 0
    rows = db.scalars(
        select(Task).where(*filters).order_by(Task.id.desc()).offset((page - 1) * page_size).limit(page_size)
    ).all()
    items = [services.serialize_task(db, t) for t in rows]
    counts = dict(db.execute(select(Task.status, func.count()).group_by(Task.status)).all())
    return {"items": items, "total": int(total), "page": page, "page_size": page_size,
            "status_counts": counts}


@app.get("/api/tasks/{task_id}", response_model=schemas.TaskDetail)
def get_task(task_id: int, db: Session = Depends(get_db)):
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return services.serialize_task(db, task, with_subtasks=True)


@app.post("/api/tasks/{task_id}/cancel", response_model=schemas.TaskDetail)
def cancel_task(task_id: int, db: Session = Depends(get_db)):
    task = services.cancel_task(db, task_id)
    return services.serialize_task(db, task, with_subtasks=True)


@app.post("/api/tasks/{task_id}/restart", response_model=schemas.TaskDetail)
def restart_task(task_id: int, db: Session = Depends(get_db)):
    task = services.restart_task(db, task_id)
    return services.serialize_task(db, task, with_subtasks=True)


@app.post("/api/workers/register")
def worker_register(body: schemas.WorkerRegister, db: Session = Depends(get_db)):
    w = services.register_worker(db, body.worker_id, body.hostname)
    return {"worker_id": w.id, "hostname": w.hostname}


@app.post("/api/workers/heartbeat")
def worker_heartbeat(body: schemas.WorkerHeartbeat, db: Session = Depends(get_db)):
    w = services.heartbeat(db, body.worker_id, body.current_subtask_id)
    cancelled = False
    if body.current_subtask_id:
        from .models import Subtask

        st = db.get(Subtask, body.current_subtask_id)
        if st:
            cancelled = services.is_task_cancelled(db, st.task_id)
    return {
        "worker_id": w.id,
        "cancel_requested": cancelled,
    }


@app.post("/api/workers/claim", response_model=schemas.ClaimResponse)
def worker_claim(body: schemas.ClaimRequest, db: Session = Depends(get_db)):
    st = services.claim_subtask(db, body.worker_id, body.hostname)
    return {"subtask": st}


@app.post("/api/workers/subtasks/{subtask_id}/progress", response_model=schemas.SubtaskOut)
def worker_progress(subtask_id: int, body: schemas.ProgressUpdate, db: Session = Depends(get_db)):
    return services.update_progress(db, subtask_id, body.worker_id, body.progress, body.log_append)


@app.post("/api/workers/subtasks/{subtask_id}/complete", response_model=schemas.SubtaskOut)
def worker_complete(subtask_id: int, body: schemas.CompleteRequest, db: Session = Depends(get_db)):
    return services.complete_subtask(
        db,
        subtask_id,
        body.worker_id,
        body.status,
        body.log_append,
        body.error_message,
        body.progress,
    )


@app.get("/api/workers/subtasks/{subtask_id}/cancel-check")
def cancel_check(subtask_id: int, db: Session = Depends(get_db)):
    from .models import Subtask

    st = db.get(Subtask, subtask_id)
    if not st:
        raise HTTPException(404, "Subtask not found")
    return {"cancelled": services.is_task_cancelled(db, st.task_id)}


# Serve built frontend if present
_STATIC = Path(__file__).resolve().parent.parent / "static"
if _STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        index = _STATIC / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(404)
