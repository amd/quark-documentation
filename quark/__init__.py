#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

try:
    from .version import __version__, git_version  # type: ignore[unused-ignore, import-not-found]
    if git_version != "unknown":
        __version__ += "+" + git_version
except ImportError:
    __version__ = 'unknown'
