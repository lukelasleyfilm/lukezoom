"""
lukezoom.core.filelock — Portable advisory file lock.

Provides a context manager that serialises read-modify-write cycles
on YAML/Markdown files.  Uses ``msvcrt`` on Windows and ``fcntl`` on
POSIX — no third-party dependencies.

Usage::

    with FileLock(path):
        data = yaml.safe_load(path.read_text())
        data["new"] = "value"
        path.write_text(yaml.dump(data))
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"


class FileLock:
    """Cross-platform advisory file lock using a ``.lock`` sidecar file.

    Parameters
    ----------
    path : Path
        The file to protect.  The lock file is ``path.lock``.
    timeout : float
        Maximum seconds to wait for the lock (default 5).
    poll : float
        Seconds between retry attempts (default 0.05).
    """

    def __init__(
        self,
        path: Path,
        timeout: float = 5.0,
        poll: float = 0.05,
    ) -> None:
        self.lock_path = Path(str(path) + ".lock")
        self.timeout = timeout
        self.poll = poll
        self._fd: Optional[int] = None

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.release()

    # -- acquire / release --------------------------------------------------

    def acquire(self) -> None:
        """Block until the lock is acquired or *timeout* expires."""
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                # O_CREAT | O_EXCL is atomic: fails if file already exists
                self._fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                return
            except FileExistsError:
                if time.monotonic() >= deadline:
                    # Stale lock recovery: if the lock file is older than
                    # 2x our timeout, assume it's stale and break it.
                    try:
                        age = time.time() - os.path.getmtime(str(self.lock_path))
                        if age > self.timeout * 2:
                            log.warning(
                                "Breaking stale lock (%.1fs old): %s",
                                age,
                                self.lock_path,
                            )
                            self._force_remove()
                            continue
                    except OSError:
                        pass
                    raise TimeoutError(
                        f"Could not acquire lock on {self.lock_path} "
                        f"within {self.timeout}s"
                    )
                time.sleep(self.poll)
            except OSError as exc:
                # On some Windows configs O_EXCL raises PermissionError
                if _IS_WINDOWS and isinstance(exc, PermissionError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Could not acquire lock on {self.lock_path} "
                            f"within {self.timeout}s"
                        ) from exc
                    time.sleep(self.poll)
                else:
                    raise

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self._force_remove()

    def _force_remove(self) -> None:
        """Best-effort removal of the lock file."""
        try:
            os.unlink(str(self.lock_path))
        except OSError:
            pass
