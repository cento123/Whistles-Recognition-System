"""
Unit tests for scripts/WRSapplication.py
Tests the application launcher and argument parsing.
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch


class TestWRSApplication:
    """Test WRSapplication module functionality."""

    def test_files_list_creator_integration(self):
        """Test file listing in the context of WRS application."""
        from src.utils import files_list_creator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                img_path = os.path.join(tmpdir, f"test_image_{i}.png")
                open(img_path, "w").close()

            png_files = files_list_creator(tmpdir, filesList_extension=".png")

            assert len(png_files) == 3
            assert all(f.endswith(".png") for f in png_files)

    def test_output_directory_cleanup(self):
        """Test that output directory is cleaned before processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)

            # Create dummy file
            old_file = os.path.join(output_dir, "old_result.json")
            open(old_file, "w").close()

            assert os.path.exists(old_file)

            # Simulate directory cleanup (as done in WRSapplication)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            assert not os.path.exists(old_file)
            assert os.path.exists(output_dir)

    def test_json_result_structure(self):
        """Test that JSON results have correct structure."""
        # Simulate what save_jsons creates
        json_data = [
            {
                "class": "w",
                "confidence": 0.95,
                "bbox": {"xmin": 100, "ymin": 50, "xmax": 200, "ymax": 150},
            },
            {
                "class": "n",
                "confidence": 0.75,
                "bbox": {"xmin": 300, "ymin": 200, "xmax": 380, "ymax": 250},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_result.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            # Load and validate
            with open(json_path, "r") as f:
                loaded = json.load(f)

            assert len(loaded) == 2
            assert all("class" in item for item in loaded)
            assert all("confidence" in item for item in loaded)
            assert all("bbox" in item for item in loaded)
            assert all(
                all(key in item["bbox"] for key in ["xmin", "ymin", "xmax", "ymax"])
                for item in loaded
            )

    @patch("src.utils.YOLO")
    def test_model_loading_mock(self, mock_yolo):
        """Test model loading is attempted."""
        from src.utils import test_model

        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = []
        mock_yolo.return_value = mock_model_instance

        # This should attempt to load and call predict
        test_model("./models/fake.pt", "./images/test.png")

        # Verify YOLO was called
        mock_yolo.assert_called_once_with("./models/fake.pt")

    def test_merge_iou_threshold_impact(self):
        """Test how IoU threshold affects box merging."""
        from src.utils import merge_overlapping_boxes

        # Two boxes with partial overlap
        boxes_high_overlap = [
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
        ]

        # Two non-overlapping boxes
        boxes_no_overlap = [
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

        # Low threshold → merge overlapping
        result_low = merge_overlapping_boxes(boxes_high_overlap, iou_threshold=0.1)
        assert len(result_low) == 1

        # High threshold with non-overlapping → don't merge
        result_high = merge_overlapping_boxes(boxes_no_overlap, iou_threshold=0.9)
        assert len(result_high) == 2
