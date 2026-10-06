from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def quiet_console(logger: logging.Logger):
    """Temporarily detach `logger`'s own handlers (its direct-to-stdout
    StreamHandler from get_logger, plus any others) for the duration of the
    `with` block, so nothing it logs reaches the console -- used around
    Colab's real (GPU) runs, where tqdm progress bars must be the only
    thing that renders in the notebook output cell. `logger` still
    propagates to the root logger as usual, so anything routed through
    `file_logging` at the same time is unaffected -- only the direct
    console handler is removed.
    """
    removed = list(logger.handlers)
    for handler in removed:
        logger.removeHandler(handler)
    try:
        yield
    finally:
        for handler in removed:
            logger.addHandler(handler)


@contextmanager
def file_logging(log_path: str | Path):
    """Route anything logged through a plain `logging.getLogger(...)` (with
    no handler of its own — e.g. training/train.py's per-run logger)
    to `log_path` instead of nowhere, for the duration of the `with` block.

    This only captures *our own* run-scoped logging (per-step loss, run
    start/end).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    previous_root_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_root_level)
        handler.close()
