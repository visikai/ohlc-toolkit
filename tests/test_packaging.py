"""The package's typing marker: present in the source tree and the wheel.

PEP 561: without a ``py.typed`` marker, a type checker treats every
``ohlc_toolkit`` import in a CONSUMER'S code as untyped and the package's
own annotations do nothing for anyone downstream. The marker is data, so
it can silently fall out of a build config; these tests pin it where the
suite can see it.
"""

from importlib import resources


def test_the_package_ships_a_py_typed_marker() -> None:
    """The PEP 561 marker sits inside the package, next to __init__."""
    marker = resources.files("ohlc_toolkit") / "py.typed"
    assert marker.is_file()
