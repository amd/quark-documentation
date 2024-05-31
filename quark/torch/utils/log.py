#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
import os
from typing import Any, Optional, Set
from logging import LogRecord


class DebugLogger:

    def __init__(self, name: str, debug_file_dir: str = 'quark_logs') -> None:
        self.logger = logging.getLogger(f'{name}_debug')
        self.logger.setLevel(logging.DEBUG)

        if not os.path.exists(debug_file_dir):
            os.makedirs(debug_file_dir)

        file_handler = logging.FileHandler(os.path.join(debug_file_dir, f'{name}_debug.log'), mode="w")
        file_handler.setLevel(logging.DEBUG)
        self.logger.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)


class CustomFormatter(logging.Formatter):
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    default_fmt = "\n[QUARK-%(levelname)s]: %(message)s"
    FORMATS = {
        logging.ERROR: RED + default_fmt + RESET,
        logging.WARNING: YELLOW + default_fmt + RESET,
        logging.INFO: GREEN + default_fmt + RESET,
    }

    def format(self, record: LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class DuplicateFilter(logging.Filter):

    def __init__(self) -> None:
        super().__init__()
        self.msgs: Set[str] = set()

    def filter(self, record: LogRecord) -> bool:
        allow_duplicate = getattr(record, 'allow_duplicate', False)
        if allow_duplicate or record.msg not in self.msgs:
            self.msgs.add(record.msg)
            return True
        return False


class ScreenLogger:

    def __init__(self, name: str):
        self.logger = logging.getLogger(f'{name}_screen')
        self.logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.propagate = False
        console_formatter = CustomFormatter()
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(console_handler)

    def info(self, msg: str, allow_duplicate: bool = True, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, extra={'allow_duplicate': allow_duplicate}, *args, **kwargs)

    def warning(self, msg: str, allow_duplicate: bool = True, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, extra={'allow_duplicate': allow_duplicate}, *args, **kwargs)

    def error(self,
              msg: str,
              error_code: Optional[str] = None,
              allow_duplicate: bool = True,
              *args: Any,
              **kwargs: Any) -> None:
        if error_code:
            msg = f"[Error Code: {error_code}] {msg}"
        self.logger.error(msg, extra={'allow_duplicate': allow_duplicate}, *args, **kwargs)
        exit(1)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)
