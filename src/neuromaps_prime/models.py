"""Models for resources in the neuromaps_prime graph."""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, field_validator


class Resource(BaseModel, ABC):
    """Base model for resources in the neuromaps_prime graph."""

    name: str
    description: str
    file_path: Path

    @abstractmethod
    def fetch(self) -> Path:
        """Fetch the resource."""
        pass

    # later add validation if the file path is valid, exists, etc.


class SurfaceAtlas(Resource):
    """Model for surface atlas resources."""

    space: str
    density: str
    hemisphere: str
    resource_type: str

    @field_validator("hemisphere")
    def validate_hemisphere(cls, v: str) -> str:
        """Validate hemisphere values."""
        valid = ["left", "right"]
        if v not in valid:
            raise ValueError(f"Hemisphere must be one of {valid}")
        return v

    def fetch(self) -> Path:
        """Fetch the surface resource."""
        return self.file_path

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        return f"{self.name}"


class SurfaceTransform(Resource):
    """Model for surface transform resources."""

    source_space: str
    target_space: str
    density: str
    hemisphere: str
    resource_type: str

    @field_validator("hemisphere")
    def validate_hemisphere(cls, v: str) -> str:
        """Validate hemisphere values."""
        valid = ["left", "right"]
        if v not in valid:
            raise ValueError(f"Hemisphere must be one of {valid}")
        return v

    def fetch(self) -> Path:
        """Fetch the transform resource."""
        return self.file_path

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        return self.name


class VolumeAtlas(Resource):
    """Model for volume atlas resources."""

    space: str
    resolution: str
    resource_type: str

    def fetch(self) -> Path:
        """Fetch the volume resource."""
        return self.file_path

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        return self.name


class VolumeTransform(Resource):
    """Model for volume transform resources."""

    source_space: str
    target_space: str
    resolution: str
    resource_type: str

    def fetch(self) -> Path:
        """Fetch the transform resource."""
        return self.file_path

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        return self.name
