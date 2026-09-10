"""Validate complete LAS/LAZ results and cache validation in a sidecar receipt.

Receipts bind to sizes and mtimes of both files. They are a completion cache,
not a cryptographic guarantee against silent storage corruption.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import laspy
import numpy as np


class InvalidLasResult(ValueError):
    pass


def fingerprint(path: Path) -> dict:
    st = path.stat()
    if not path.is_file():
        raise InvalidLasResult(f'OUTPUT_INVALID: not a regular file: {path}')
    return dict(path=str(path.absolute()), size=st.st_size, mtime_ns=st.st_mtime_ns)


def code_identity(script: Path) -> dict:
    """Record exact source/model hashes, without unpickling a model in the worker."""
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):
                h.update(block)
        return h.hexdigest()
    root=script.parent.parent
    files=[script] + sorted(root.glob('*.py')) + sorted(script.parent.glob('*.py'))
    hashes={str(p.relative_to(root)):sha(p) for p in files if p.is_file()}
    model=Path(os.getenv('RF_REFINER','')) if os.getenv('RF_REFINER') else root/'models'/'etalon_rf_scene_20260909.pkl'
    if model.is_file():
        hashes['rf_model']=sha(model)
    return dict(source='worker_run',script=script.name,sha256=hashes)


def _receipt_path(output: Path) -> Path:
    return output.with_name(output.name+'.tls-complete.json')


def validate_result(source: Path, output: Path, provenance: dict | None = None, checkpoint=None) -> bool:
    """False only when output is missing; malformed existing output is an error.

    Legacy outputs are streamed once and checked against source coordinates.
    Existing invalid files are never removed or overwritten here.
    """
    try:
        out_stat=fingerprint(output)
    except FileNotFoundError:
        return False
    in_stat=fingerprint(source)
    if source.resolve()==output.resolve():
        raise InvalidLasResult('OUTPUT_EQUALS_INPUT: use a separate result path')
    receipt_path=_receipt_path(output)
    receipt=None
    try:
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    except (FileNotFoundError,ValueError):
        pass
    cached=(isinstance(receipt,dict) and receipt.get('schema')==1
            and receipt.get('input')==in_stat and receipt.get('output')==out_stat
            and receipt.get('validation')=='full_coordinates_and_count')
    if not cached:
        try:
            with laspy.open(str(source)) as src, laspy.open(str(output)) as dst:
                expected=int(src.header.point_count)
                if dst.header.point_count != expected:
                    raise InvalidLasResult('OUTPUT_INVALID: LAS point count differs from input')
                if not np.array_equal(src.header.scales,dst.header.scales) or not np.array_equal(src.header.offsets,dst.header.offsets):
                    raise InvalidLasResult('OUTPUT_INVALID: coordinate scales/offsets changed')
                seen=0
                for original in src.chunk_iterator(1_000_000):
                    if checkpoint is not None:
                        checkpoint()
                    result=dst.read_points(len(original))
                    if len(result)!=len(original):
                        raise InvalidLasResult('OUTPUT_INVALID: truncated point records')
                    if any(not np.array_equal(original[d],result[d]) for d in ('X','Y','Z')):
                        raise InvalidLasResult('OUTPUT_INVALID: point coordinates or order differ')
                    seen+=len(result)
                if seen!=expected:
                    raise InvalidLasResult('OUTPUT_INVALID: incomplete input or result point data')
        except OSError:
            raise  # worker retries transient storage failures
        except InvalidLasResult:
            raise
        except Exception as exc:
            raise InvalidLasResult(f'OUTPUT_INVALID: LAS cannot be fully read: {exc}') from exc
        if fingerprint(source)!=in_stat or fingerprint(output)!=out_stat:
            raise OSError('Files changed during LAS verification; retry after writes finish')
    if cached and provenance is None:
        return True
    receipt=dict(schema=1,input=in_stat,output=out_stat,
                 validation='full_coordinates_and_count',
                 provenance=provenance or {'source':'legacy_verified','algorithm':'unknown'})
    tmp=receipt_path.with_name(receipt_path.name+f'.{os.getpid()}.writing')
    try:
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(receipt,f,indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,receipt_path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return True
