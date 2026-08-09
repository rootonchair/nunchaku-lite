import os
import re

import setuptools


def get_base_version(root_dir: str) -> str:
    version_path = os.path.join(root_dir, "src", "nunchaku_lite", "__version__.py")
    with open(version_path, encoding="utf-8") as version_file:
        contents = version_file.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', contents, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find __version__ assignment in {version_path}")
    return match.group(1)


if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    version = os.getenv("NUNCHAKU_LITE_RELEASE_VERSION") or get_base_version(root_dir)

    setuptools.setup(
        version=version,
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src", include=["nunchaku_lite", "nunchaku_lite.*"]),
    )
