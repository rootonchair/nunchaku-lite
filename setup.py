import os
import re

import setuptools
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
_VERSION_FILE = os.path.join("nunchaku_lite", "__version__.py")


def get_base_version(root_dir: str) -> str:
    version_path = os.path.join(root_dir, "src", _VERSION_FILE)
    with open(version_path, encoding="utf-8") as version_file:
        contents = version_file.read()
    match = _VERSION_RE.search(contents)
    if not match:
        raise RuntimeError(f"Could not find __version__ assignment in {version_path}")
    return match.group(1)


def _stamp_version(path: str, version: str) -> None:
    """Write the release version into a built copy of ``__version__.py`` (never the source checkout), so the
    installed package reports the version it was released as."""

    with open(path, encoding="utf-8") as version_file:
        contents = version_file.read()
    contents = _VERSION_RE.sub(f'__version__ = "{version}"', contents, count=1)
    os.remove(path)  # the build tree may hard-link the source file; replace it instead of writing through
    with open(path, "w", encoding="utf-8") as version_file:
        version_file.write(contents)


class BuildPy(build_py):
    def run(self):
        super().run()
        target = os.path.join(self.build_lib, _VERSION_FILE)
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", _VERSION_FILE)
        # editable installs import the source tree directly: nothing to stamp, and never touch the source file
        if getattr(self, "editable_mode", False) or not os.path.exists(target):
            return
        if os.path.realpath(target) == os.path.realpath(source):
            return
        _stamp_version(target, self.distribution.get_version())


class SDist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        # an sdist built for a release keeps its version when pip later builds a wheel from it without the env var
        _stamp_version(os.path.join(base_dir, "src", _VERSION_FILE), self.distribution.get_version())


if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    version = os.getenv("NUNCHAKU_LITE_RELEASE_VERSION") or get_base_version(root_dir)

    setuptools.setup(
        version=version,
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src", include=["nunchaku_lite", "nunchaku_lite.*"]),
        cmdclass={"build_py": BuildPy, "sdist": SDist},
    )
