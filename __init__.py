# ruff: noqa: N999 - the module name is the clone directory, e.g. custom_nodes/nunchaku-lite
"""ComfyUI custom-node entry point for Nunchaku Lite.

Clone this repository into ``ComfyUI/custom_nodes/`` and install its dependencies into ComfyUI's Python
environment (``pip install -e .`` from the clone, or ``pip install nunchaku-lite``). ComfyUI imports this file
and registers the nodes from :mod:`nunchaku_lite.comfyui.nodes`. The checkout's ``src`` comes first on the path, so
the nodes always run with the library version they were written for.

This file is not part of the ``nunchaku_lite`` package. Outside ComfyUI (for example when pytest imports the
repository root as a package) it registers no nodes.
"""

import importlib.util
import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if importlib.util.find_spec("folder_paths") is not None:  # ComfyUI's top-level module: we run inside ComfyUI
    _SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if os.path.isdir(os.path.join(_SRC, "nunchaku_lite")) and _SRC not in sys.path:
        sys.path.insert(0, _SRC)

    from nunchaku_lite.comfyui.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
