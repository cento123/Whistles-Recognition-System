"""
Pytest configuration and shared fixtures for WRS tests.
"""

import os
import tempfile

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test usage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_bbox():
    """Provide a sample bounding box for testing."""
    return {"xmin": 100, "ymin": 50, "xmax": 200, "ymax": 150}


@pytest.fixture
def sample_detection():
    """Provide a sample detection dictionary."""
    return {
        "class": "w",
        "confidence": 0.95,
        "bbox": {"xmin": 100, "ymin": 50, "xmax": 200, "ymax": 150},
    }


@pytest.fixture
def overlapping_detections():
    """Provide overlapping detection dictionaries for merge testing."""
    return [
        {
            "class": "w",
            "confidence": 0.85,
            "bbox": {"xmin": 100, "ymin": 50, "xmax": 200, "ymax": 150},
        },
        {
            "class": "w",
            "confidence": 0.92,
            "bbox": {"xmin": 150, "ymin": 70, "xmax": 250, "ymax": 180},
        },
        {
            "class": "n",
            "confidence": 0.75,
            "bbox": {"xmin": 300, "ymin": 200, "xmax": 380, "ymax": 250},
        },
    ]


@pytest.fixture
def contained_detections():
    """Provide detections where one is contained in another."""
    return [
        {
            "class": "w",
            "confidence": 0.80,
            "bbox": {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 300},
        },
        {
            "class": "w",
            "confidence": 0.90,
            "bbox": {"xmin": 150, "ymin": 150, "xmax": 250, "ymax": 250},
        },
    ]


@pytest.fixture
def config_yaml_path(temp_dir):
    """Create a sample config.yaml for testing."""
    config_content = """
Fbin: 46.875
Tbin: 0.04
Foffset: 0.0
Npxs: 512
"""
    config_path = os.path.join(temp_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path
