#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any, List, Optional
import torch
from torch.utils.cpp_extension import load, _get_build_directory
import os
import time
import platform
from quark.torch.utils.log import ScreenLogger

logger = ScreenLogger(__name__)
path = Path(__file__).parent


def compile_kernel(kernel_name: str, compile_dir: Optional[str], extra_cuda_cflags: List[str],
                   extra_cflags: List[str]) -> Any:
    os_name = platform.system()
    if os_name == "Windows":
        logger.warning("Windows system only support CPU mode, skip compiling kernel, Quark runs in CPU mode.")
    else:
        if torch.cuda.is_available():
            try:
                verbose_flag = False
                compile_dir = "" if compile_dir is None else compile_dir
                compile_dir = _get_build_directory(kernel_name, verbose_flag) if compile_dir == "" else compile_dir

                if not os.path.exists(compile_dir):
                    os.makedirs(compile_dir)

                logger.info("C++ kernel build directory " + compile_dir)
                return load(name=kernel_name,
                            sources=[
                                str(path / "csrc/python_function_export.cpp"),
                                str(path / "csrc/fake_tensor.cpp"),
                                str(path / "csrc/fake_tensor_cuda_hip.cu"),
                                str(path / "csrc/quant_dequant_fp8.cpp"),
                                str(path / "csrc/quant_dequant_fp8_cuda_hip.cu"),
                                str(path / "csrc/fake_quant_fp8_cuda.cu"),
                                str(path / "csrc/fake_quant_fp8.cpp")
                            ],
                            build_directory=compile_dir,
                            extra_cuda_cflags=extra_cuda_cflags,
                            extra_cflags=extra_cflags,
                            verbose=verbose_flag)
            except Exception as e:
                logger.error("C++ kernel compile error\n" + str(e) +
                             "\n\nPlease check if CUDA_HOME or ROCM_HOME is set.")
        else:
            logger.warning("Using torch-cpu, Skip compiling kernel, Quark runs in CPU mode.")
    return None


logger.info("C++ kernel compilation check start.")
is_cuda_runtime = 1
if torch.version.cuda:
    is_cuda_runtime = 1
else:
    is_cuda_runtime = 0

extra_cuda_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
extra_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
if torch.cuda.is_available():
    if is_cuda_runtime == 1:
        extra_cuda_cflags.extend(["-O2", "--extended-lambda"])
    else:
        extra_cuda_cflags.extend(["-O2"])

compile_dir = None
kernel_name = "kernel_ext"
is_python_module = True

start_time = time.time()
kernel_ext = compile_kernel(kernel_name, compile_dir, extra_cuda_cflags, extra_cflags)
end_time = time.time()
execution_time = end_time - start_time
logger.info(
    "C++ kernel compilation is already complete. Ending the C++ kernel compilation check. Total time: {:.4f} seconds".
    format(execution_time))
