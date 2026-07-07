"""Module associated with remote storages."""

from .github import GitHubStorage
from .osf import OSFStorage

__all__ = ["GitHubStorage", "OSFStorage"]
