"""MLX runtime tests: collected only where MLX is installed (Apple silicon)."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("mlx") else ["test_*.py"]
