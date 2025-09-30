
import pytest
from pathlib import Path
import os

from neuromaps_nhp.resources.fetch_resource import fetch_atlas, fetch_transform, fetch_resource, search_resources
from neuromaps_nhp.config import config

def test_fetch_atlas():
    # Test fetching a known atlas resource
    atlas_path = fetch_atlas(
        resource_name="sphere",
        source="Yerkes19",
        density="32k",
        hemisphere="left"
    )
    assert atlas_path is not None
    assert os.path.exists(atlas_path)

def test_fetch_atlas_invalid():
    # Test fetching an invalid atlas resource
    atlas_path = fetch_atlas(
        resource_name="invalid_resource",
        source="Yerkes19",
        density="32k",
        hemisphere="left"
    )
    assert atlas_path is None

def test_fetch_transform():
    # Test fetching a known transform resource
    transform_path = fetch_transform(
        resource_name="sphere",
        source="Yerkes19",
        target="D99",
        density="32k",
        hemisphere="left"
    )
    assert transform_path is not None
    assert os.path.exists(transform_path)

def test_fetch_transform_invalid():
    # Test fetching an invalid transform resource
    transform_path = fetch_transform(
        resource_name="invalid_resource",
        source="Yerkes19",
        target="D99",
        density="32k",
        hemisphere="left"
    )
    assert transform_path is None


def test_fetch_resource_atlas():
    # Test fetching a known resource using the generic fetch_resource function
    resource_path = fetch_resource(
        resource_type="atlas",
        source="Yerkes19",
        density="32k",
        hemisphere="left"
    )
    assert resource_path is not None
    assert os.path.exists(resource_path)

def test_fetch_resource_transform():
    # Test fetching a known resource using the generic fetch_resource function
    resource_path = fetch_resource(
        resource_type="transform",
        source="Yerkes19",
        target="D99",
        density="32k",
        hemisphere="left",
        resource_name="sphere"
    )
    assert resource_path is not None
    assert os.path.exists(resource_path)
    
def test_fetch_resource_invalid():
    # Test fetching an invalid resource using the generic fetch_resource function
    resource_path = fetch_resource(
        resource_type="invalid_type",
        source="Yerkes19",
        density="32k",
        hemisphere="left"
    )
    assert resource_path is None

