# Release Wheels

Release wheels are built by GitHub Actions with `cibuildwheel` when a version
tag is pushed.

Before releasing, make sure `src/nunchaku_lite/__version__.py` points at the
release series. For example, `0.1.0dev` is valid for tag `v0.1.0`.

## Trigger A Release Build

Create and push a tag from the commit you want to release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow validates that the tag matches the committed package
version after allowing a trailing `dev` postfix. Tag `v0.1.0` may build from
either `0.1.0` or `0.1.0dev`. For tag builds, CI strips only the build version
by setting `NUNCHAKU_LITE_RELEASE_VERSION`; it does not edit the committed
source file.

The root `nunchaku_lite` wheel is pure Python and uses the package version
directly, for example:

```text
0.1.0
```

After release, bump to the next dev version:

```bash
git add src/nunchaku_lite/__version__.py
git commit -m "Start 0.1.1 development"
```

Manual builds are available from the `Release Wheels` workflow through
`workflow_dispatch`; manual builds may use dev versions for validation.

## Build Matrix

The release workflow builds the root Python package. CUDA kernels are provided
by Hugging Face `kernels` by default or by installing the vendored
`./nunchaku-lite-kernels` package separately.

Local CUDA kernel package builds still support `NUNCHAKU_INSTALL_MODE=FAST` and
`NUNCHAKU_INSTALL_MODE=ALL`.

## Local Reproduction

Install `build` and create the root Python wheel. Set
`NUNCHAKU_LITE_RELEASE_VERSION` only when reproducing a tag release; omit it
for normal dev builds.

```bash
python -m pip install build

export NUNCHAKU_LITE_RELEASE_VERSION=0.1.0

python -m build --wheel
```

To build the local CUDA kernels package from source:

```bash
NUNCHAKU_INSTALL_MODE=ALL pip install ./nunchaku-lite-kernels
```
