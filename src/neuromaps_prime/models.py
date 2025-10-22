"""Models for resources in the neuromaps_prime surface graph."""

from pathlib import Path

from pydantic import BaseModel, field_validator


class ResourceModel(BaseModel):
    """Base model for resources in the surface graph."""

    name: str
    description: str
    file_path: Path

    # later add validation if the file path is valid, exists, etc.


class SurfaceAtlasModel(ResourceModel):
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


class SurfaceTransformModel(ResourceModel):
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


class VolumeAtlasModel(ResourceModel):
    """Model for volume atlas resources."""

    space: str
    resolution: str
    resource_type: str


class VolumeTransformModel(ResourceModel):
    """Model for volume transform resources."""

    source_space: str
    target_space: str
    resolution: str
    resource_type: str
