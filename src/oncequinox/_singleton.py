"""Defines a metaclass for singleton Equinox modules."""

__all__ = ("SingletonModuleMeta",)


import weakref
from typing import Any

import equinox as eqx

ModuleMeta: type[type[eqx.Module]] = type(eqx.Module)

_singleton_insts: weakref.WeakKeyDictionary[type, object] = weakref.WeakKeyDictionary()


class SingletonModuleMeta(ModuleMeta):  # type: ignore[misc]
    """A metaclass for singleton Equinox modules.

    This metaclass ensures that only one instance of a class exists. Multiple calls
    to instantiate the class will return the same instance. The singleton instances
    are stored using weak references, allowing them to be garbage collected when no
    strong references remain.

    Note:
        When a singleton instance already exists, any arguments passed to subsequent
        instantiation calls are ignored, and the existing instance is returned.

    Examples:
        Create a basic singleton module:

        >>> import equinox as eqx
        >>> import oncequinox as oqx
        >>> class Config(eqx.Module, metaclass=oqx.SingletonModuleMeta):
        ...     value: int = 42
        >>> config1 = Config()
        >>> config2 = Config()
        >>> config1 is config2
        True

        Singletons are maintained per class:

        >>> class ConfigA(eqx.Module, metaclass=oqx.SingletonModuleMeta):
        ...     name: str = "A"
        >>> class ConfigB(eqx.Module, metaclass=oqx.SingletonModuleMeta):
        ...     name: str = "B"
        >>> config_a = ConfigA()
        >>> config_b = ConfigB()
        >>> config_a is config_b
        False

        Subclasses have their own singleton instances:

        >>> class BaseConfig(eqx.Module, metaclass=oqx.SingletonModuleMeta):
        ...     value: int = 1
        >>> class DerivedConfig(BaseConfig):
        ...     value: int = 2
        >>> base = BaseConfig()
        >>> derived = DerivedConfig()
        >>> base is derived
        False
        >>> base.value
        1
        >>> derived.value
        2

        Arguments to ``__init__`` are only used for the first instantiation:

        >>> class HasValue(eqx.Module, metaclass=oqx.SingletonModuleMeta):
        ...     value: int
        ...     def __init__(self, value: int):
        ...         self.value = value
        >>> v1 = HasValue(10)
        >>> v1.value
        10
        >>> v2 = HasValue(20)  # Args ignored, returns existing instance
        >>> v2.value
        10
        >>> v1 is v2
        True

    """

    def __call__(cls, /, *args: Any, **kwargs: Any) -> Any:
        # Check if instance already exists
        if cls in _singleton_insts:
            return _singleton_insts[cls]
        # Create new instance and cache it
        self = super().__call__(*args, **kwargs)
        _singleton_insts[cls] = self
        return self
