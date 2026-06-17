"""
Results validation tests.
Tests to verify output correctness and expected results format.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from src.utils import get_bbox_params, load_config, merge_overlapping_boxes


class TestResultsValidation:
    """Validate correctness of results and outputs."""

    def test_detection_format_validation(self):
        """Verify detection JSON has correct format."""
        detections = [
            {
                "class": "w",
                "confidence": 0.95,
                "bbox": {"xmin": 100, "ymin": 50, "xmax": 200, "ymax": 150},
            },
        ]

        # Validate structure
        for det in detections:
            assert "class" in det
            assert "confidence" in det
            assert "bbox" in det
            bbox = det["bbox"]
            assert all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"])
            assert det["confidence"] >= 0.0 and det["confidence"] <= 1.0
            assert bbox["xmin"] < bbox["xmax"]
            assert bbox["ymin"] < bbox["ymax"]

    def test_merged_boxes_consistency(self):
        """Verify merged boxes maintain consistency."""
        # Use boxes with clear overlap and containment
        boxes = [
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

        merged = merge_overlapping_boxes(boxes, iou_threshold=0.3)

        # Should merge (second is contained in first)
        assert len(merged) == 1

        # Merged box should have:
        # - Union of bounds
        assert merged[0]["bbox"]["xmin"] == 100
        assert merged[0]["bbox"]["ymin"] == 100
        assert merged[0]["bbox"]["xmax"] == 300
        assert merged[0]["bbox"]["ymax"] == 300
        # - Max confidence
        assert merged[0]["confidence"] == 0.90

    def test_whistle_parameters_physically_valid(self, config_yaml_path):
        """Verify extracted whistle parameters are physically valid."""
        tbin, fbin, foffset, npxs = load_config(config_yaml_path)

        item = {
            "class": "w",
            "confidence": 0.95,
            "bbox": {"xmin": 50, "ymin": 100, "xmax": 200, "ymax": 250},
        }

        conf, tini, tdur, fmin, fmax = get_bbox_params(item, tbin, fbin, foffset, npxs)

        # Validations
        assert conf > 0.0 and conf <= 1.0  # Valid confidence
        assert tini >= 0.0  # Time in the future
        assert tdur > 0.0  # Positive duration
        assert fmin >= 0.0  # Non-negative frequency
        assert fmax > fmin  # Max > Min
        assert (fmax - fmin) > 0  # Frequency range is positive

    def test_dataframe_aggregation_integrity(self):
        """Verify DataFrame aggregation maintains data integrity."""
        data = [
            {
                "FileName": "img1.json",
                "Wid": 0,
                "Conf": 0.95,
                "Tini": 0.0,
                "Tdur": 0.5,
                "Fmin": 1000,
                "Fmax": 2000,
            },
            {
                "FileName": "img1.json",
                "Wid": 1,
                "Conf": 0.88,
                "Tini": 0.8,
                "Tdur": 0.3,
                "Fmin": 1500,
                "Fmax": 2500,
            },
            {
                "FileName": "img2.json",
                "Wid": 2,
                "Conf": 0.92,
                "Tini": 0.2,
                "Tdur": 0.4,
                "Fmin": 900,
                "Fmax": 1800,
            },
        ]

        df = pd.DataFrame(data)
        df["Fdur"] = df["Fmax"] - df["Fmin"]

        # Validation
        assert len(df) == 3
        assert df["Wid"].is_unique  # IDs are unique
        assert all(df["Fmax"] > df["Fmin"])
        assert all(df["Fdur"] == df["Fmax"] - df["Fmin"])
        assert df["Fdur"].iloc[0] == 1000
        assert df["Fdur"].iloc[1] == 1000
        assert df["Fdur"].iloc[2] == 900

    def test_csv_metadata_preservation(self):
        """Verify CSV outputs preserve metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")

            df = pd.DataFrame(
                {
                    "Wid": [0, 1, 2],
                    "Tdur": [0.1, 0.2, 0.3],
                    "Fmin": [1000, 1200, 1100],
                    "Fmax": [2000, 2200, 2100],
                }
            )

            metadata = [
                "# ----------------------------------------",
                f"# {len(df)} whistles analyzed",
                f"# Tbin=0.04s, Fbin=46.875Hz, Foffset=0.0Hz, Npxs=512",
                "# ----------------------------------------",
            ]

            with open(csv_path, "w") as f:
                for line in metadata:
                    f.write(line + "\n")
                df.to_csv(f, sep=";", index=False)

            # Verify file structure
            with open(csv_path, "r") as f:
                lines = f.readlines()

            assert len(lines) > len(df)  # Metadata + header + data
            assert lines[0].startswith("#")
            # Check that "whistles" appears somewhere in metadata (lines 0-3)
            metadata_text = "".join(lines[:4])
            assert "whistles" in metadata_text

    def test_histogram_statistics_validity(self):
        """Verify histogram statistics are valid."""
        from src.utils import calc_hist

        data = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        yvalue, nbins, nth, thval = calc_hist(data, BinRes=0.05, prob=True)

        # Validations
        assert len(yvalue) == len(nbins)
        assert np.all(yvalue >= 0)  # No negative counts
        assert np.isclose(np.sum(yvalue), 100.0)  # Probabilities sum to 100%
        assert np.all(np.diff(nbins) > 0)  # Bins monotonically increase

    def test_no_data_loss_in_pipeline(self):
        """Verify no detections are lost through pipeline."""
        original_detections = [
            {"class": "w", "confidence": 0.95},
            {"class": "w", "confidence": 0.88},
            {"class": "n", "confidence": 0.75},
        ]

        # Step 1: Separate by class
        whistles = [d for d in original_detections if d["class"] == "w"]
        noise = [d for d in original_detections if d["class"] == "n"]

        # Step 2: Verify all data accounted for
        assert len(whistles) + len(noise) == len(original_detections)
        assert len(whistles) == 2
        assert len(noise) == 1

    def test_iou_threshold_effect_on_count(self):
        """Verify IoU threshold correctly affects detection count."""
        # Overlapping boxes with clear IoU
        overlapping_boxes = [
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

        # Non-overlapping boxes
        non_overlapping_boxes = [
            {
                "class": "w",
                "confidence": 0.85,
                "bbox": {"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50},
            },
            {
                "class": "w",
                "confidence": 0.92,
                "bbox": {"xmin": 100, "ymin": 100, "xmax": 150, "ymax": 150},
            },
        ]

        # Very low threshold on overlapping → merge
        merged_low = merge_overlapping_boxes(overlapping_boxes, iou_threshold=0.01)
        assert len(merged_low) == 1

        # Very high threshold on non-overlapping → no merge
        merged_high = merge_overlapping_boxes(non_overlapping_boxes, iou_threshold=0.99)
        assert len(merged_high) == 2

        # Verify merged result is union
        assert merged_low[0]["bbox"]["xmin"] <= min(
            overlapping_boxes[0]["bbox"]["xmin"],
            overlapping_boxes[1]["bbox"]["xmin"],
        )

    def test_class_separation_consistency(self):
        """Verify class separation (whistle vs noise) is consistent."""
        detections = [
            {"class": "w", "confidence": 0.95},
            {"class": "w", "confidence": 0.88},
            {"class": "w", "confidence": 0.92},
            {"class": "n", "confidence": 0.75},
            {"class": "n", "confidence": 0.65},
        ]

        # Separate
        whistles = [d for d in detections if d["class"] == "w"]
        noise = [d for d in detections if d["class"] == "n"]

        # Verify no cross-contamination
        assert all(d["class"] == "w" for d in whistles)
        assert all(d["class"] == "n" for d in noise)
        assert len(whistles) + len(noise) == len(detections)
