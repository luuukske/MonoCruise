"""Registry semantics: register is exclusive, replace is the swap the watchdog needs."""
from __future__ import annotations

import pytest

from core.thread_management.registry import Registry


class _Stub:
    def __init__(self, name: str, tag: str = ""):
        self.name = name
        self.tag = tag


@pytest.fixture
def reg():
    return Registry()


def test_register_then_get(reg):
    t = _Stub("worker")
    reg.register(t)
    assert reg.get_thread("worker") is t
    assert reg.all_threads() == [t]


def test_registering_the_same_name_twice_raises(reg):
    """Silent overwrite would leave two live threads fighting over one name."""
    reg.register(_Stub("worker"))
    with pytest.raises(KeyError):
        reg.register(_Stub("worker"))


def test_replace_swaps_without_needing_an_unregister(reg):
    """The watchdog restart path: old out, new in, one lock acquisition."""
    old, new = _Stub("worker", "old"), _Stub("worker", "new")
    reg.register(old)
    reg.replace(new)
    assert reg.get_thread("worker") is new
    assert reg.all_threads() == [new]


def test_replace_works_on_an_unknown_name(reg):
    new = _Stub("worker")
    reg.replace(new)
    assert reg.get_thread("worker") is new


def test_missing_thread_raises_keyerror(reg):
    """Callers catch KeyError to survive a missing sibling; it must not be None."""
    with pytest.raises(KeyError):
        reg.get_thread("nope")


def test_unregister_is_idempotent(reg):
    reg.register(_Stub("worker"))
    reg.unregister("worker")
    reg.unregister("worker")
    with pytest.raises(KeyError):
        reg.get_thread("worker")


def test_objects_and_threads_share_one_namespace(reg):
    reg.register(_Stub("worker"))
    with pytest.raises(KeyError):
        reg.register_object("worker", object())

    marker = object()
    reg.register_object("panel", marker)
    assert reg.get("panel") is marker
    assert "panel" in reg.names()
    with pytest.raises(KeyError):
        reg.get_thread("panel")


def test_get_prefers_threads_over_objects(reg):
    t = _Stub("shared")
    reg.replace(t)
    reg._objects["shared"] = object()
    assert reg.get("shared") is t
