# Release Wheels

Releases are built by GitHub Actions when a version tag is pushed: an sdist and
a pure-Python wheel, attached to a GitHub Release and published to PyPI as
[`nunchaku-lite`](https://pypi.org/project/nunchaku-lite/).

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

The build also writes the release version into the packaged copy of
`__version__.py` (wheel and sdist), so `nunchaku_lite.__version__` reports the
released version; the checkout is unchanged. The root `nunchaku_lite` wheel is
pure Python and uses the package version directly, for example:

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

## PyPI Publishing

The `Publish to PyPI` job uploads the tag's sdist and wheel with
[trusted publishing](https://docs.pypi.org/trusted-publishers/): no API token
is stored in the repository. One-time setup:

1. On PyPI, add a trusted publisher for project `nunchaku-lite` (a pending
   publisher before the first upload): owner `rootonchair`, repository
   `nunchaku-lite`, workflow `wheels.yml`, environment `pypi`.
2. In the GitHub repository settings, create the `pypi` environment
   (optionally with required reviewers, to approve each upload).

A tag push then runs lint, version validation, the build and `twine check`,
and publishes. PyPI versions are immutable: a broken release is fixed with a
new version, not a re-upload.

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
