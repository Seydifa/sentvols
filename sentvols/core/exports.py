from __future__ import annotations

import importlib
import pkgutil
from collections import defaultdict


_EXPORT_REGISTRY = defaultdict(dict)
_CORE_MODULES_LOADED = False
_PUBLIC_MODULES = ("explainers", "features", "models", "plots", "portfolio", "utils")


def _load_core_modules():
    global _CORE_MODULES_LOADED

    if _CORE_MODULES_LOADED:
        return

    core_package_name = __name__.rsplit(".", 1)[0]
    core_package = importlib.import_module(core_package_name)

    for module_info in pkgutil.iter_modules(core_package.__path__):
        if module_info.name.startswith("_") or module_info.name == "exports":
            continue

        importlib.import_module(f"{core_package_name}.{module_info.name}")

    _CORE_MODULES_LOADED = True


def registration(module, name=None):
    if module not in _PUBLIC_MODULES:
        available_modules = ", ".join(_PUBLIC_MODULES)
        raise ValueError(
            f"Module '{module}' is not exportable. Available modules: {available_modules}."
        )

    def decorator(obj):
        export_name = name or obj.__name__
        existing = _EXPORT_REGISTRY[module].get(export_name)

        if existing is not None and existing is not obj:
            raise ValueError(
                f"Export '{export_name}' is already registered in module '{module}'."
            )

        _EXPORT_REGISTRY[module][export_name] = obj
        return obj

    return decorator


def registre(module, name=None):
    return registration(module=module, name=name)


def get_module_export_by_name(name):
    _load_core_modules()

    if name not in _PUBLIC_MODULES:
        available_modules = ", ".join(_PUBLIC_MODULES)
        raise ValueError(
            f"Module '{name}' not found. Available modules: {available_modules}."
        )

    return dict(_EXPORT_REGISTRY.get(name, {}))


def get_registered_export(module, name):
    exports = get_module_export_by_name(module)

    try:
        return exports[name]
    except KeyError as exc:
        raise AttributeError(
            f"Module '{module}' has no registered export named '{name}'."
        ) from exc


def list_registered_exports(module):
    return sorted(get_module_export_by_name(module))


def export_public_namespace(module):
    exports = get_module_export_by_name(module)
    return {"__all__": sorted(exports), **exports}
