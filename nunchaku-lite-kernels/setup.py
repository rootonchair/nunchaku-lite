import os
import re
import subprocess
import sys

import setuptools
import torch
from packaging import version as packaging_version
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            ext.extra_compile_args.setdefault("cxx", [])
            ext.extra_compile_args.setdefault("nvcc", [])
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args["cxx"] += ext.extra_compile_args["msvc"]
                ext.extra_compile_args["nvcc"] += ext.extra_compile_args["nvcc_msvc"]
            else:
                ext.extra_compile_args["cxx"] += ext.extra_compile_args["gcc"]
        super().build_extensions()


def get_sm_targets() -> list[str]:
    nvcc_path = os.path.join(CUDA_HOME, "bin/nvcc") if CUDA_HOME else "nvcc"
    try:
        nvcc_output = subprocess.check_output([nvcc_path, "--version"]).decode()
        match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", nvcc_output)
        if match is None:
            raise RuntimeError("nvcc version not found")
        nvcc_version = match.group(2)
        print(f"Found nvcc version: {nvcc_version}")
    except Exception as exc:
        raise RuntimeError("nvcc not found") from exc

    support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
    support_sm121 = packaging_version.parse(nvcc_version) >= packaging_version.parse("13.0")

    install_mode = os.getenv("NUNCHAKU_INSTALL_MODE", "FAST")
    if install_mode == "FAST":
        targets = []
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            sm = f"{capability[0]}{capability[1]}"
            if sm == "120" and support_sm120:
                sm = "120a"
            if sm == "121" and support_sm121:
                sm = "121a"
            if sm not in ["75", "80", "86", "89", "120a", "121a"]:
                raise RuntimeError(f"Unsupported SM {sm}")
            if sm not in targets:
                targets.append(sm)
    else:
        if install_mode != "ALL":
            raise RuntimeError("NUNCHAKU_INSTALL_MODE must be FAST or ALL")
        targets = ["75", "80", "86", "89"]
        if support_sm120:
            targets.append("120a")
        if support_sm121:
            targets.append("121a")
    return targets


if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    os.chdir(root_dir)
    base_version = (
        os.getenv("NUNCHAKU_LITE_KERNELS_RELEASE_VERSION")
        or os.getenv("NUNCHAKU_LITE_RELEASE_VERSION")
        or "0.1.0dev"
    )

    if torch.version.cuda is None:
        raise RuntimeError("nunchaku_lite_kernels requires a CUDA-enabled PyTorch build")
    torch_version = torch.__version__.split("+")[0]
    torch_major_minor_version = ".".join(torch_version.split(".")[:2])
    version = f"{base_version}+cu{torch.version.cuda}torch{torch_major_minor_version}"

    def here(path: str) -> str:
        return os.path.join(root_dir, path)

    sm_targets = get_sm_targets()
    print(f"Detected SM targets: {sm_targets}", file=sys.stderr)

    gcc_flags = ["-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-fvisibility=hidden", "-g", "-std=c++20", "-UNDEBUG", "-Og"]
    msvc_flags = ["/DENABLE_BF16=1", "/DBUILD_NUNCHAKU=1", "/std:c++20", "/UNDEBUG", "/Zc:__cplusplus", "/FS"]
    nvcc_threads = os.getenv("NUNCHAKU_NVCC_THREADS", str(len(sm_targets)))
    nvcc_flags = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-g",
        "-std=c++20",
        "-UNDEBUG",
        "-Xcudafe",
        "--diag_suppress=20208",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        f"--threads={nvcc_threads}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    if os.getenv("NUNCHAKU_BUILD_WHEELS", "0") == "0":
        nvcc_flags.append("--generate-line-info")
    for target in sm_targets:
        nvcc_flags += ["-gencode", f"arch=compute_{target},code=sm_{target}"]

    extension = CUDAExtension(
        name="nunchaku_lite_kernels._C",
        sources=[
            "nunchaku_lite_kernels_cuda/pybind.cpp",
            "nunchaku_lite_kernels_cuda/torch_dispatch.cpp",
            "nunchaku_lite_kernels_cuda/interop/torch.cpp",
            "nunchaku_lite_kernels_cuda/kernels/zgemm/gemm_w4a4.cu",
            "nunchaku_lite_kernels_cuda/kernels/zgemm/gemm_w4a4_launch_fp16_int4.cu",
            "nunchaku_lite_kernels_cuda/kernels/zgemm/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
            "nunchaku_lite_kernels_cuda/kernels/zgemm/gemm_w4a4_launch_fp16_fp4.cu",
            "nunchaku_lite_kernels_cuda/kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu",
            "nunchaku_lite_kernels_cuda/kernels/zgemm/gemm_w4a4_launch_bf16_fp4.cu",
            "nunchaku_lite_kernels_cuda/kernels/zgemm/attention.cu",
            "nunchaku_lite_kernels_cuda/kernels/awq/gemm_awq.cu",
            "nunchaku_lite_kernels_cuda/kernels/awq/gemv_awq.cu",
            "nunchaku_lite_kernels_cuda/kernels/modulation.cu",
            "nunchaku_lite_kernels_cuda/kernels/cross_head_qk_norm_rope.cu",
        ],
        extra_compile_args={
            "gcc": gcc_flags,
            "msvc": msvc_flags,
            "nvcc": nvcc_flags,
            "nvcc_msvc": ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/FS", "-Xcompiler", "/bigobj"],
        },
        include_dirs=[here("nunchaku_lite_kernels_cuda")],
    )

    setuptools.setup(
        name="nunchaku_lite_kernels",
        version=version,
        description="Local CUDA kernel package for nunchaku_lite.",
        package_dir={"": "torch-ext"},
        packages=setuptools.find_packages(where="torch-ext", include=["nunchaku_lite_kernels"]),
        ext_modules=[extension],
        cmdclass={"build_ext": CustomBuildExtension},
    )
