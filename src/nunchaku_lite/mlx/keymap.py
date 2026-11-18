"""Map checkpoint (Diffusers) module names to MLX model paths.

Nunchaku Lite checkpoints and runtime manifests name modules after the Diffusers model. mflux re-implements each
model with its own module tree, but ships declarative Diffusers -> mflux weight mappings (``WeightTarget`` lists);
:meth:`PrefixMap.from_mflux` turns one into a module-prefix map so manifest targets and dense weights land on the
right mflux modules without a hand-written table.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

__all__ = ["PrefixMap"]

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _compile(pattern: str) -> tuple[re.Pattern[str], tuple[str, ...]]:
    names: list[str] = []
    out, pos = [], 0
    for match in _PLACEHOLDER.finditer(pattern):
        out.append(re.escape(pattern[pos : match.start()]))
        name = match.group(1)
        out.append(f"(?P={name})" if name in names else f"(?P<{name}>\\d+)")
        names.append(name)
        pos = match.end()
    out.append(re.escape(pattern[pos:]))
    return re.compile("".join(out)), tuple(dict.fromkeys(names))


class PrefixMap:
    """Rename module prefixes: the longest dotted ancestor of a name that matches a rule is replaced.

    Rules are ``(source, target)`` module patterns that may contain ``{placeholder}`` integer fields, e.g.
    ``("transformer_blocks.{block}.attn.to_out.0", "transformer_blocks.{block}.attn.to_out")``. Names that match
    no rule are returned unchanged, so the map applies equally to module paths (``...attn.to_out.0``) and
    parameter keys (``...attn.to_out.0.qweight``).
    """

    def __init__(self, rules: Iterable[tuple[str, str]] = ()):
        self._exact: dict[str, str] = {}
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for source, target in rules:
            if source == target:
                continue
            regex, names = _compile(source)
            if not names:
                self._exact[source] = target
            elif set(_PLACEHOLDER.findall(target)) <= set(names):
                self._patterns.append((regex, target))

    @classmethod
    def from_mflux(cls, weight_targets: Iterable[Any]) -> PrefixMap:
        """Build a map from mflux ``WeightTarget`` s (``to_pattern`` / ``from_pattern`` with a shared leaf)."""

        rules = []
        for target in weight_targets:
            to_module, _, to_leaf = target.to_pattern.rpartition(".")
            for source in target.from_pattern:
                from_module, _, from_leaf = source.rpartition(".")
                if from_leaf == to_leaf and from_module and to_module and from_module != to_module:
                    rules.append((from_module, to_module))
        return cls(dict.fromkeys(rules))

    def __bool__(self) -> bool:
        return bool(self._exact or self._patterns)

    def _match(self, head: str) -> str | None:
        if head in self._exact:
            return self._exact[head]
        for regex, target in self._patterns:
            match = regex.fullmatch(head)
            if match is not None:
                return target.format(**match.groupdict())
        return None

    def __call__(self, name: str) -> str:
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            mapped = self._match(".".join(parts[:i]))
            if mapped is not None:
                return ".".join([mapped, *parts[i:]])
        return name
