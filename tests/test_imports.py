import subprocess
import sys

import pytest


def test_public_packages_import_without_legacy_graph_modules():
    import RetailPilot
    import RetailPilot.api
    import RetailPilot.graphs

    assert RetailPilot.graph is RetailPilot.graphs.graph


@pytest.mark.parametrize("first", ["RetailPilot", "ecommerce_agent"])
def test_legacy_imports_share_canonical_modules(first):
    # Fresh interpreters exercise both import orders without cached modules.
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import sys

first = {first!r}
other = 'ecommerce_agent' if first == 'RetailPilot' else 'RetailPilot'
for suffix in ('', '.config', '.database', '.agent', '.api', '.graphs'):
    left = importlib.import_module(first + suffix)
    right = importlib.import_module(other + suffix)
    assert left is right, suffix
    assert left.__name__ == 'RetailPilot' + suffix, suffix
    assert left.__spec__.name == 'RetailPilot' + suffix, suffix
assert sys.modules['ecommerce_agent.config'].settings is sys.modules['RetailPilot.config'].settings
assert sys.modules['ecommerce_agent.agent'].graph is sys.modules['RetailPilot.graphs'].graph
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
