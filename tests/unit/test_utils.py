"""
Unit tests for src/utils.py functions.
Tests box merging, IoU calculation, file listing, and histogram computation.
"""

import os

import numpy as np
import pytest

from src.utils import (
    calc_hist,
    calc_iou,
    files_list_creator,
    get_bbox_params,
    is_contained,
    load_config,
    merge_overlapping_boxes,
    merge_two_boxes,
)


class TestBoxMerging:
    """Test bounding box merging functions."""

    def test_merge_two_boxes_basic(self, sample_detection):
        """Test merging two identical detections."""
        det1 = sample_detection.copy()
        det2 = sample_detection.copy()
        det2["confidence"] = 0.99

        result = merge_two_boxes(det1, det2)

        assert result["class"] == "w"
        assert result["confidence"] == 0.99  # max of 0.95 and 0.99
        assert result["bbox"]["xmin"] == 100
        assert result["bbox"]["xmax"] == 200

    def test_merge_two_boxes_different_bounds(self):
        """Test merging boxes with different bounds creates union."""
        det1 = {
            "class": "w",
            "confidence": 0.80,
            "bbox": {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200},
        }
        det2 = {
            "class": "w",
            "confidence": 0.90,
            "bbox": {"xmin": 150, "ymin": 150, "xmax": 250, "ymax": 250},
        }

        result = merge_two_boxes(det1, det2)

        assert result["bbox"]["xmin"] == 100  # min of 100, 150
        assert result["bbox"]["ymin"] == 100  # min of 100, 150
        assert result["bbox"]["xmax"] == 250  # max of 200, 250
        assert result["bbox"]["ymax"] == 250  # max of 200, 250


class TestIOUCalculation:
    """Test IoU (Intersection over Union) calculation."""

    def test_calc_iou_identical_boxes(self, sample_bbox):
        """Test IoU of identical boxes is 1.0."""
        iou = calc_iou(sample_bbox, sample_bbox)
        assert iou == pytest.approx(1.0)

    def test_calc_iou_non_overlapping(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        box1 = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        box2 = {"xmin": 20, "ymin": 20, "xmax": 30, "ymax": 30}
        iou = calc_iou(box1, box2)
        assert iou == 0.0

    def test_calc_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        box1 = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        box2 = {"xmin": 5, "ymin": 5, "xmax": 15, "ymax": 15}

        iou = calc_iou(box1, box2)

        # Overlap area: 5x5 = 25
        # Union area: 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.143
        assert iou == pytest.approx(25 / 175, rel=1e-3)

    def test_calc_iou_one_inside_other(self):
        """Test IoU when one box is completely inside another."""
        outer = {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}
        inner = {"xmin": 20, "ymin": 20, "xmax": 80, "ymax": 80}

        iou = calc_iou(outer, inner)

        # Overlap = inner area = 3600
        # Union = outer area = 10000
        # IoU = 3600/10000 = 0.36
        assert iou == pytest.approx(0.36)


class TestContainment:
    """Test box containment checking."""

    def test_is_contained_fully_inside(self):
        """Test detection of fully contained box."""
        inner = {"xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150}
        outer = {"xmin": 0, "ymin": 0, "xmax": 200, "ymax": 200}
        assert is_contained(inner, outer) is True

    def test_is_contained_not_inside(self):
        """Test detection when boxes don't overlap."""
        box1 = {"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50}
        box2 = {"xmin": 100, "ymin": 100, "xmax": 150, "ymax": 150}
        assert is_contained(box1, box2) is False

    def test_is_contained_partial_overlap(self):
        """Test detection with significant overlap but not contained."""
        inner = {"xmin": 75, "ymin": 75, "xmax": 175, "ymax": 175}
        outer = {"xmin": 0, "ymin": 0, "xmax": 150, "ymax": 150}
        # Inner is 10000, outer is 22500, intersection is 5625
        # containment_ratio = 5625/10000 = 0.5625 > 0.5 → should be True
        assert is_contained(inner, outer) is True


class TestMergeOverlappingBoxes:
    """Test merging of overlapping boxes."""

    def test_merge_single_box(self, sample_detection):
        """Test that single box is returned unchanged."""
        result = merge_overlapping_boxes([sample_detection])
        assert len(result) == 1
        assert result[0] == sample_detection

    def test_merge_empty_list(self):
        """Test that empty list is handled correctly."""
        result = merge_overlapping_boxes([])
        assert result == []

    def test_merge_non_overlapping(self):
        """Test that non-overlapping boxes remain separate."""
        boxes = [
            {
                "class": "w",
                "confidence": 0.80,
                "bbox": {"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50},
            },
            {
                "class": "w",
                "confidence": 0.90,
                "bbox": {"xmin": 100, "ymin": 100, "xmax": 150, "ymax": 150},
            },
        ]
        result = merge_overlapping_boxes(boxes, iou_threshold=0.3)
        assert len(result) == 2

    def test_merge_overlapping_boxes(self, overlapping_detections):
        """Test merging of overlapping detections."""
        result = merge_overlapping_boxes(overlapping_detections, iou_threshold=0.3)
        # Two whistle boxes should merge, noise box remains separate
        assert len(result) == 2
        # Merged box should have highest confidence
        whistles = [box for box in result if box["class"] == "w"]
        assert len(whistles) == 1
        assert whistles[0]["confidence"] == 0.92

    def test_merge_contained_boxes(self, contained_detections):
        """Test merging when one box contains another."""
        result = merge_overlapping_boxes(contained_detections)
        # Should merge the contained box with the outer
        assert len(result) == 1
        assert result[0]["confidence"] == 0.90  # max confidence
        # Merged box should be the union (outer bounds)
        assert result[0]["bbox"]["xmin"] == 100
        assert result[0]["bbox"]["xmax"] == 300


class TestFilesListCreator:
    """Test file listing functionality."""

    def test_files_list_creator_empty_folder(self, temp_dir):
        """Test with empty folder."""
        result = files_list_creator(temp_dir)
        assert result == []

    def test_files_list_creator_no_extension_filter(self, temp_dir):
        """Test listing all files without extension filter."""
        # Create test files
        open(os.path.join(temp_dir, "file1.txt"), "w").close()
        open(os.path.join(temp_dir, "file2.json"), "w").close()

        result = files_list_creator(temp_dir)
        assert len(result) == 2

    def test_files_list_creator_with_extension(self, temp_dir):
        """Test filtering by extension."""
        open(os.path.join(temp_dir, "file1.txt"), "w").close()
        open(os.path.join(temp_dir, "file2.json"), "w").close()

        result = files_list_creator(temp_dir, filesList_extension=".json")
        assert len(result) == 1
        assert result[0].endswith(".json")

    def test_files_list_creator_with_substring(self, temp_dir):
        """Test filtering by filename substring."""
        open(os.path.join(temp_dir, "test_file1.txt"), "w").close()
        open(os.path.join(temp_dir, "other_file.txt"), "w").close()

        result = files_list_creator(temp_dir, filesList_contains="test")
        assert len(result) == 1
        assert "test" in result[0]

    def test_files_list_creator_recursive(self, temp_dir):
        """Test recursive subdirectory searching."""
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        open(os.path.join(temp_dir, "file1.txt"), "w").close()
        open(os.path.join(subdir, "file2.txt"), "w").close()

        result = files_list_creator(temp_dir, filesList_extension=".txt")
        assert len(result) == 2


class TestCalcHist:
    """Test histogram calculation."""

    def test_calc_hist_empty_data(self):
        """Test histogram with empty data."""
        yvalue, nbins, nth, thval = calc_hist([], 1.0)
        assert len(yvalue) == 0
        assert len(nbins) == 0
        assert nth is None
        assert thval is None

    def test_calc_hist_single_value(self):
        """Test histogram with single value."""
        data = [5.0]
        yvalue, nbins, nth, thval = calc_hist(data, 1.0)
        assert len(yvalue) == 6  # bins from 0.5 to 5.5
        assert np.sum(yvalue) == 1  # one sample

    def test_calc_hist_uniform_distribution(self):
        """Test histogram with uniform distribution."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        yvalue, nbins, nth, thval = calc_hist(data, 1.0)
        assert len(yvalue) == 5
        assert np.sum(yvalue) == 5  # five samples

    def test_calc_hist_probability_mode(self):
        """Test histogram in probability mode."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        yvalue, nbins, nth, thval = calc_hist(data, 1.0, prob=True)
        assert np.isclose(np.sum(yvalue), 100.0)  # percentages sum to 100

    def test_calc_hist_with_threshold(self):
        """Test histogram with cumulative threshold."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        yvalue, nbins, nth, thval = calc_hist(data, 1.0, prob=True, Ythreshold=50.0)
        assert nth is not None
        assert thval is not None
        assert thval >= 50.0


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_config_valid(self, config_yaml_path):
        """Test loading valid configuration."""
        tbin, fbin, foffset, npxs = load_config(config_yaml_path)

        assert tbin == 0.04
        assert fbin == 46.875
        assert foffset == 0.0
        assert npxs == 512

    def test_load_config_file_not_found(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


class TestGetBboxParams:
    """Test bounding box parameter extraction."""

    def test_get_bbox_params_basic(self):
        """Test extraction of time and frequency parameters."""
        item = {
            "bbox": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
            "confidence": 0.95,
        }
        tbin, fbin, foffset, npxs = 0.04, 46.875, 0.0, 512

        conf, tini, tdur, fmin, fmax = get_bbox_params(item, tbin, fbin, foffset, npxs)

        assert conf == 0.95
        assert tini == 0.0  # xmin * tbin = 0 * 0.04
        assert tdur == 4.0  # (xmax - xmin) * tbin = 100 * 0.04
        assert fmin == 0.0 + (512 - 100) * 46.875  # Foffset + (Npxs - ymax) * Fbin
        assert fmax == 0.0 + (512 - 0) * 46.875  # Foffset + (Npxs - ymin) * Fbin

    def test_get_bbox_params_with_offset(self):
        """Test extraction with frequency offset."""
        item = {
            "bbox": {"xmin": 10, "ymin": 50, "xmax": 50, "ymax": 150},
            "confidence": 0.85,
        }
        tbin, fbin, foffset, npxs = 0.04, 50.0, 1000.0, 512

        conf, tini, tdur, fmin, fmax = get_bbox_params(item, tbin, fbin, foffset, npxs)

        assert conf == 0.85
        assert tini == 0.4  # 10 * 0.04
        assert tdur == 1.6  # (50 - 10) * 0.04
        assert fmin == 1000.0 + (512 - 150) * 50.0
        assert fmax == 1000.0 + (512 - 50) * 50.0
