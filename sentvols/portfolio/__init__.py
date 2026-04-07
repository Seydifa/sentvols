from ..core import (
    export_public_namespace,
    get_registered_export,
    list_registered_exports,
)


_NAMESPACE = export_public_namespace("portfolio")
globals().update(_NAMESPACE)
__all__ = _NAMESPACE["__all__"]


def __getattr__(name):
    return get_registered_export("portfolio", name)


def __dir__():
    return sorted(set(globals()) | set(list_registered_exports("portfolio")))
