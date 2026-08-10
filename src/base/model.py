"""Lifecycle management for models that own expensive resources."""

from __future__ import annotations

import abc


class ManagedModel(abc.ABC):
    """Base class for models that can be loaded and unloaded explicitly.

    Subclasses own the actual model/resource reference.  They should acquire
    it in :meth:`_load` and release it in :meth:`_unload`.
    """

    def __init__(self) -> None:
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying resource is currently loaded."""

        return self._is_loaded

    def load(self) -> None:
        """Load the underlying resource once."""

        if self._is_loaded:
            return

        self._load()
        self._is_loaded = True

    def unload(self) -> None:
        """Release the underlying resource if it is loaded."""

        if not self._is_loaded:
            return

        self._unload()
        self._is_loaded = False

    def __enter__(self) -> ManagedModel:
        """Load the model and return it for use inside a ``with`` block."""

        self.load()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Unload the model when leaving a ``with`` block."""

        self.unload()

    @abc.abstractmethod
    def _load(self) -> None:
        """Acquire the model's underlying resource."""

    @abc.abstractmethod
    def _unload(self) -> None:
        """Release the model's underlying resource and clear its references."""
