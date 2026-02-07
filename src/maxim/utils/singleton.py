"""Singleton registry for managing global instances.

Eliminates repetitive get_X/set_X/init_X patterns across the codebase
by providing a centralized, thread-safe singleton management system.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class SingletonRegistry:
    """Thread-safe registry for managing global singleton instances.

    Usage:
        # Define a singleton
        _output_manager = SingletonRegistry[AgentOutputManager]("output_manager")

        # Get/set/init
        manager = _output_manager.get()
        _output_manager.set(my_manager)
        manager = _output_manager.init(AgentOutputManager, instance_id="foo")

        # Or use factory for lazy init
        _output_manager.set_factory(lambda: AgentOutputManager(instance_id="default"))
        manager = _output_manager.get(create_if_missing=True)
    """

    _instances: dict[str, Any] = {}
    _factories: dict[str, Callable[[], Any]] = {}
    _lock = threading.Lock()

    def __init__(self, name: str) -> None:
        """Create a singleton slot with the given name.

        Args:
            name: Unique identifier for this singleton.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the singleton name."""
        return self._name

    def get(self, create_if_missing: bool = False) -> T | None:
        """Get the singleton instance.

        Args:
            create_if_missing: If True and a factory is registered,
                create the instance if it doesn't exist.

        Returns:
            The singleton instance, or None if not set.
        """
        with self._lock:
            instance = self._instances.get(self._name)

            if instance is None and create_if_missing:
                factory = self._factories.get(self._name)
                if factory is not None:
                    instance = factory()
                    self._instances[self._name] = instance

            return instance

    def get_or_raise(self) -> T:
        """Get the singleton instance, raising if not set.

        Returns:
            The singleton instance.

        Raises:
            RuntimeError: If the singleton is not initialized.
        """
        instance = self.get()
        if instance is None:
            raise RuntimeError(f"Singleton '{self._name}' not initialized")
        return instance

    def set(self, instance: T) -> None:
        """Set the singleton instance.

        Args:
            instance: The instance to store.
        """
        with self._lock:
            self._instances[self._name] = instance

    def set_factory(self, factory: Callable[[], T]) -> None:
        """Set a factory for lazy initialization.

        Args:
            factory: Callable that creates the instance when needed.
        """
        with self._lock:
            self._factories[self._name] = factory

    def init(self, cls: type[T], *args: Any, **kwargs: Any) -> T:
        """Initialize the singleton with the given class and arguments.

        Args:
            cls: The class to instantiate.
            *args: Positional arguments for the constructor.
            **kwargs: Keyword arguments for the constructor.

        Returns:
            The created instance.
        """
        instance = cls(*args, **kwargs)
        self.set(instance)
        return instance

    def clear(self) -> None:
        """Clear the singleton instance."""
        with self._lock:
            self._instances.pop(self._name, None)

    def is_initialized(self) -> bool:
        """Check if the singleton is initialized."""
        with self._lock:
            return self._name in self._instances

    @classmethod
    def clear_all(cls) -> None:
        """Clear all singleton instances (useful for testing)."""
        with cls._lock:
            cls._instances.clear()
            cls._factories.clear()


# Type-safe generic version for better IDE support
class Singleton(Generic[T]):
    """Type-safe singleton wrapper.

    Usage:
        _manager: Singleton[AgentOutputManager] = Singleton("output_manager")

        # Type-safe access
        manager = _manager.get()  # Returns AgentOutputManager | None
        _manager.set(my_manager)  # Type-checked
    """

    def __init__(self, name: str) -> None:
        self._registry = SingletonRegistry(name)

    @property
    def name(self) -> str:
        return self._registry.name

    def get(self, create_if_missing: bool = False) -> T | None:
        return self._registry.get(create_if_missing)

    def get_or_raise(self) -> T:
        return self._registry.get_or_raise()

    def set(self, instance: T) -> None:
        self._registry.set(instance)

    def set_factory(self, factory: Callable[[], T]) -> None:
        self._registry.set_factory(factory)

    def init(self, cls: type[T], *args: Any, **kwargs: Any) -> T:
        return self._registry.init(cls, *args, **kwargs)

    def clear(self) -> None:
        self._registry.clear()

    def is_initialized(self) -> bool:
        return self._registry.is_initialized()


# Convenience functions for simple use cases
def get_singleton(name: str) -> Any | None:
    """Get a singleton by name."""
    return SingletonRegistry(name).get()


def set_singleton(name: str, instance: Any) -> None:
    """Set a singleton by name."""
    SingletonRegistry(name).set(instance)


def init_singleton(name: str, cls: type[T], *args: Any, **kwargs: Any) -> T:
    """Initialize a singleton by name."""
    return SingletonRegistry(name).init(cls, *args, **kwargs)


__all__ = [
    "Singleton",
    "SingletonRegistry",
    "get_singleton",
    "set_singleton",
    "init_singleton",
]
