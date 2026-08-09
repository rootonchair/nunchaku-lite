# Installation

## Prerequisites

- Python 3.10 or newer
- PyTorch 2.7 or newer with CUDA
- Diffusers 0.36 or newer

Local kernel package builds also require CUDA toolkit 12.6 or newer with
`nvcc`. The default root install uses prebuilt kernels from
`rootonchair/nunchaku-lite-kernels` through the Hugging Face `kernels` library
instead of compiling kernels locally.

!!! note "CUDA version compatibility"

    The root `nunchaku_lite` package does not compile CUDA code. Use a CUDA
    toolkit with `nvcc` only when installing `./nunchaku-lite-kernels`. CUDA
    12.6 or newer is the documented minimum; Blackwell `sm120a` requires CUDA
    12.8 or newer, and `sm121a` requires CUDA 13.0 or newer.

## Install From Source

Clone the repository and install from the repository root:

```bash
git clone https://github.com/rootonchair/nunchaku_lite.git
cd nunchaku_lite
pip install .
```

This installs the Python dependencies from `pyproject.toml` and builds the
root Python package without compiling CUDA code. At runtime, `nunchaku_lite`
uses Hugging Face prebuilt kernels by default.

## Install Local CUDA Kernels

To compile and use local CUDA kernels from the vendored kernel package:

```bash
pip install ./nunchaku-lite-kernels
```

At runtime, `nunchaku_lite` prefers a locally installed
`nunchaku_lite_kernels` package. If it is not installed, the op wrappers load
`rootonchair/nunchaku-lite-kernels` with `kernels.get_kernel(..., version=2)`.

The Hugging Face kernel package lists support for CUDA `7.5`, `8.0`, `8.6`,
`8.9`, `12.0a`, and `12.1a`.

To build all supported GPU architectures:

```bash
NUNCHAKU_INSTALL_MODE=ALL pip install ./nunchaku-lite-kernels
```

Supported targets are `sm75`, `sm80`, `sm86`, `sm89`, `sm120a`, and `sm121a`,
subject to the installed CUDA toolkit version. CUDA 12.6 or newer supports the
non-Blackwell targets, CUDA 12.8 or newer is required for `sm120a`, and CUDA
13.0 or newer is required for `sm121a`.

## Nix Kernel Builds

`nunchaku-lite-kernels` includes Nix build support through Hugging Face
kernel-builder:

```bash
nix flake show path:./nunchaku-lite-kernels
```

## Install From GitHub

```bash
pip install git+https://github.com/rootonchair/nunchaku_lite.git
```

## Build A Wheel

```bash
python setup.py bdist_wheel
pip install dist/nunchaku_lite-*.whl
```
