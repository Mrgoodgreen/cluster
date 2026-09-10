from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_storage_roots(raw: str) -> dict[str, Path]:
    """Parse STORAGE_ROOTS=display=/local/path,other=/other/path.

    Also accepts legacy display:/local/path (colon) for simple labels without IPs.
    """
    roots: dict[str, Path] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            display, local = part.split("=", 1)
        elif ":" in part:
            display, local = part.split(":", 1)
        else:
            p = Path(part).resolve()
            roots[p.name or str(p)] = p
            continue
        display = display.strip()
        local = local.strip()
        if not display or not local:
            continue
        roots[display] = Path(local).resolve()
    return roots


@dataclass(frozen=True)
class Settings:
    database_url: str
    storage_roots: dict[str, Path]
    host: str
    port: int
    cors_origins: list[str]
    worker_lease_minutes: int
    log_text_max_chars: int


def load_settings() -> Settings:
    db = os.getenv("DATABASE_URL", "sqlite:///./data/cluster.db")
    roots_raw = os.getenv(
        "STORAGE_ROOTS",
        "storage=/mnt/storage",
    )
    origins = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]
    return Settings(
        database_url=db,
        storage_roots=_parse_storage_roots(roots_raw),
        host=os.getenv("MANAGER_HOST", "0.0.0.0"),
        # Default 5357 — keep free of RAD/CS TextureEnhancement on 5257
        port=int(os.getenv("MANAGER_PORT", "5357")),
        cors_origins=origins,
        worker_lease_minutes=max(5, int(os.getenv("WORKER_LEASE_MINUTES", "30"))),
        log_text_max_chars=max(32_000, int(os.getenv("LOG_TEXT_MAX_CHARS", str(512 * 1024)))),
    )


settings = load_settings()
