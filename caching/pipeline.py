from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

class PipelineCache(metaclass=ABCMeta):
    """A cache for storing intermediate results in a pipeline."""

    @abstractmethod
    async def get(self, key: str) -> Any:
        """Get a value from the cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        pass

    @abstractmethod
    async def delete(self , key: str) -> None:
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear the cache."""
        pass

    @abstractmethod
    async def child(self,  name : str) -> PipelineCache:
        """Get a child cache."""
        pass


