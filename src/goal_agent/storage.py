"""Atomic run records and cross-process exclusive execution locks."""

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .models import AgentError


_io_lock = threading.RLock()


def write_json(path: Path, data: dict) -> None:
    # Windows readers may temporarily prevent replacing an open file. Serialize
    # in-process polling with writes and tolerate short external sharing locks.
    with _io_lock:
        pending = path.with_suffix(path.suffix + ".tmp")
        with pending.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(8):
            try:
                pending.replace(path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.005 * 2 ** min(attempt, 4))


def read_json(path: Path) -> dict:
    with _io_lock:
        if path.stat().st_size > 32_000_000:
            raise AgentError("Run record is too large.")
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AgentError("Expected a JSON object.")
    return data


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".execution.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise AgentError("This run is already executing in another process.") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
