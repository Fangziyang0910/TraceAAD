"""TraceAAD packages and lazily loaded legacy search entry points."""

from importlib import import_module

__all__ = ["Config", "TraceAAD", "Program"]


def __getattr__(name):
    modules = {"Config": ".base", "TraceAAD": ".base", "Program": ".base"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
