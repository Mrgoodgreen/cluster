from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    input_path: str = Field(..., description="Display storage path to input folder")
    output_path: str = Field(..., description="Display storage path to output folder")


class SubtaskOut(BaseModel):
    id: int
    task_id: int
    filename: str
    relative_path: str
    input_file: str
    output_file: str
    status: str
    worker_id: str | None
    started_at: datetime | None
    finished_at: datetime | None
    progress: float
    log_text: str
    error_message: str | None

    model_config = {"from_attributes": True}


class TaskOut(BaseModel):
    id: int
    uid: str
    status: str
    input_path: str
    output_path: str
    created_at: datetime
    finished_at: datetime | None
    error_message: str | None
    subtask_total: int = 0
    subtask_done: int = 0

    model_config = {"from_attributes": True}


class TaskDetail(TaskOut):
    subtasks: list[SubtaskOut] = []


class TaskList(BaseModel):
    items: list[TaskOut]
    total: int
    page: int
    page_size: int
    status_counts: dict[str, int] = Field(default_factory=dict)


class StorageNode(BaseModel):
    title: str
    key: str
    path: str
    is_leaf: bool = False
    children: list[StorageNode] | None = None


class StorageMkdirRequest(BaseModel):
    path: str = Field(..., description="Display storage path of folder to create")


class StorageMkdirResponse(BaseModel):
    path: str
    created: bool
    existed: bool = False


class WorkerRegister(BaseModel):
    worker_id: str
    hostname: str = ""


class WorkerHeartbeat(BaseModel):
    worker_id: str
    current_subtask_id: int | None = None


class ClaimRequest(BaseModel):
    worker_id: str
    hostname: str = ""


class ClaimResponse(BaseModel):
    subtask: SubtaskOut | None = None


class ProgressUpdate(BaseModel):
    worker_id: str
    progress: float = 0.0
    log_append: str = ""


class CompleteRequest(BaseModel):
    worker_id: str
    status: str  # success | error | skipped | cancelled
    log_append: str = ""
    error_message: str | None = None
    progress: float | None = None
