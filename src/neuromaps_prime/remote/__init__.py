"""Module associated with remote storages."""

from .gin import GINStorage
from .github import GitHubStorage
from .osf import OSFStorage

__all__ = ["GINStorage", "GitHubStorage", "OSFStorage"]
