"""Legacy imports alias RetailPilot modules without duplicating runtime state."""

import sys
from collections.abc import Sequence
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, spec_from_loader
from types import ModuleType


class _LegacyLoader(Loader):
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name

    def exec_module(self, module: ModuleType) -> None:
        sys.modules[module.__name__] = import_module(self.canonical_name)


class _LegacyFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if not fullname.startswith("ecommerce_agent."):
            return None
        canonical_name = "RetailPilot" + fullname[len("ecommerce_agent") :]
        canonical_spec = find_spec(canonical_name)
        if canonical_spec is None:
            return None
        return spec_from_loader(
            fullname,
            _LegacyLoader(canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


sys.meta_path.insert(0, _LegacyFinder())
sys.modules[__name__] = import_module("RetailPilot")
