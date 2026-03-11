# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import sys


logger = logging.getLogger()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class PlainTextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return _ANSI_ESCAPE_RE.sub("", formatted)


def _configure_noisy_loggers() -> None:
    # Root logger is INFO for TorchTitan, but these third-party libraries are
    # too chatty and make diff-oriented logs hard to read.
    for name in ("filelock", "httpcore", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _should_add_file_handler() -> bool:
    rank = os.environ.get("RANK")
    return rank in (None, "0")


def _maybe_add_file_handler() -> None:
    log_file = os.environ.get("TORCHTITAN_LOG_FILE")
    if not log_file:
        return
    if not _should_add_file_handler():
        return

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    # Keep file logs diff-friendly by stripping the timestamp/logger prefix and
    # removing ANSI escape sequences while leaving terminal colors intact.
    fh.setFormatter(PlainTextFormatter("%(message)s"))
    logger.addHandler(fh)


def init_logger() -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    _configure_noisy_loggers()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    _maybe_add_file_handler()

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


_logged: set[str] = set()


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)
