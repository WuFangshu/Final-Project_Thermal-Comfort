# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbinian Poeppel

import os
from typing import Sequence, Union
import logging
import time
import random
import torch
from torch.utils.cpp_extension import load as _load

LOGGER = logging.getLogger(__name__)

def defines_to_cflags(defines=Union[dict[str, Union[int, str]], Sequence[tuple[str, Union[str, int]]]]):
    cflags = []
    if isinstance(defines, dict):
        defines = defines.items()
    for key, val in defines:
        cflags.append(f"-D{key}={str(val)}")
    return cflags

curdir = os.path.dirname(__file__)

if torch.cuda.is_available():
    from packaging import version
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(device_type="cuda")[-1])[0], "lib"
        )
    else:
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0], "lib"
        )

EXTRA_INCLUDE_PATHS = (
    tuple(os.environ["XLSTM_EXTRA_INCLUDE_PATHS"].split(":"))
    if "XLSTM_EXTRA_INCLUDE_PATHS" in os.environ else ()
)

if "CONDA_PREFIX" in os.environ:
    from pathlib import Path
    from packaging import version
    import glob

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        EXTRA_INCLUDE_PATHS += tuple(
            map(str, (Path(os.environ["CONDA_PREFIX"]) / "targets").glob("**/include/"))
        )[:1]

def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    suffix = ""
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {"float": "f", "__half": "h", "__nv_bfloat16": "b", "true": "1", "false": "0"}
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    # 添加 include 和 compile 参数
    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    ]
    for eip in EXTRA_INCLUDE_PATHS:
        extra_cflags.append("-isystem")
        extra_cflags.append(eip)

    extra_ldflags = [f"/LIBPATH:{os.environ['CUDA_LIB']}", "cublas.lib"]

    myargs = {
        "verbose": True,
        "with_cuda": True,
        "extra_cflags": extra_cflags,
        "extra_cuda_cflags": [
            "-gencode", "arch=compute_86,code=sm_86",  # RTX 4060 = Ada Lovelace 架构
            "-Xptxas", "-v",
            "--use_fast_math",
            "-O3",
            "--extra-device-vectorization",
            *extra_cflags,
            *extra_cuda_cflags,
        ],
        "extra_ldflags": extra_ldflags,
    }

    myargs.update(**kwargs)
    time.sleep(random.random() * 5)
    LOGGER.info(f"Before compilation and loading of {name}.")
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")
    return mod
