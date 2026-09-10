#!/usr/bin/env python3
"""Create minimal fake .las files for mock worker smoke tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent / "storage"
IN = ROOT / "in"
OUT = ROOT / "out"


def main() -> None:
    IN.mkdir(parents=True, exist_ok=True)
    (IN / "sub").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for path in (IN / "a.las", IN / "sub" / "b.las"):
        path.write_bytes(b"FAKE_LAS_FOR_SMOKE_" + path.name.encode("ascii"))

    # Pre-existing output for skip demo on a second run; first run uses empty out/
    print(f"Prepared smoke storage under {ROOT}")
    print("  in/a.las, in/sub/b.las")
    print("  out/ (empty)")


if __name__ == "__main__":
    main()
