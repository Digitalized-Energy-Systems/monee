"""help() on any public monee.express / monee.problem callable must survive a
cp1252 Windows console (regression: U+2208 in controllable_backup_lines and
U+03A3 in create_economic_dispatch_problem crashed there)."""

import inspect
import pathlib
import pydoc

import pytest

import monee
import monee.express
import monee.problem

SRC = pathlib.Path(monee.__file__).resolve().parent


def _public_callables():
    for module in (monee.express, monee.problem):
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj):
                yield pytest.param(obj, id=f"{module.__name__}.{name}")


@pytest.mark.parametrize("obj", _public_callables())
def test_help_renders_on_cp1252(obj):
    text = pydoc.render_doc(obj, renderer=pydoc.plaintext)
    text.encode("cp1252")
    if inspect.isclass(obj):
        for _, member in inspect.getmembers(obj):
            doc = inspect.getdoc(member) or ""
            doc.encode("cp1252")


def test_src_tree_is_ascii():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(ord(c) > 127 for c in line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}")
    assert not offenders, (
        "non-ASCII characters in src/monee (crashes help()/print on cp1252 "
        "consoles): " + ", ".join(offenders[:20])
    )
