"""
Integration tests for complete WRS pipeline.
Tests end-to-end workflow: application → results analysis.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from src.utils import (
    calc_hist,
    files_list_creator,
    get_bbox_params,
    load_config,
    merge_overlapping_boxes,
)


class TestPipelineIntegration:
    """Test complete WRS pipeline workflows."""

    def test_json_to_csv_pipeline(self, config_yaml_path):
        """Test complete pipeline from JSON results to CSV analysis."""
        tbin, fbin, foffset, npxs = load_config(config_yaml_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create sample JSON results
            json_data = [
                {
                    "class": "w",
                    "confidence": 0.95,
                    "bbox": {"xmin": 10, "ymin": 50, "xmax": 100, "ymax": 150},
                },
                {
                    "class": "n",
                    "confidence": 0.75,
                    "bbox": {"xmin": 200, "ymin": 200, "xmax": 280, "ymax": 250},
                },
                {
                    "class": "w",
                    "confidence": 0.88,
                    "bbox": {"xmin": 120, "ymin": 100, "xmax": 200, "ymax": 200},
                },
            ]

            json_path = os.path.join(tmpdir, "detections.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            # Step 2: Process JSON to extract whistle parameters
            with open(json_path, "r") as f:
                data = json.load(f)

            results = []
            w_id = 0
            for item in data:
                if item.get("class") == "w":
                    conf, tini, tdur, fmin, fmax = get_bbox_params(
                        item, tbin, fbin, foffset, npxs
                    )
                    results.append(
                        {
                            "Wid": w_id,
                            "Conf": conf,
                            "Tini": tini,
                            "Tdur": tdur,
                            "Fmin": fmin,
                            "Fmax": fmax,
                        }
                    )
                    w_id += 1

            # Step 3: Create DataFrame and add derived columns
            df = pd.DataFrame(results)
            df["Fdur"] = df["Fmax"] - df["Fmin"]

            # Assertions
            assert len(df) == 2  # Two whistles
            assert "Fdur" in df.columns
            assert df["Conf"].iloc[0] == 0.95
            assert df["Conf"].iloc[1] == 0.88
            assert all(df["Fmax"] > df["Fmin"])

    def test_merging_then_analysis(self):
        """Test merging overlapping boxes then analyzing results."""
        # Original detections with overlaps
        detections = [
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

        # Merge overlapping
        merged = merge_overlapping_boxes(detections, iou_threshold=0.3)

        # Count whistles vs noise
        whistles = [box for box in merged if box["class"] == "w"]
        noise = [box for box in merged if box["class"] == "n"]

        assert len(whistles) == 1  # Merged
        assert len(noise) == 1
        assert whistles[0]["confidence"] == 0.92  # Max confidence

    def test_histogram_analysis_workflow(self):
        """Test histogram generation for statistical analysis."""
        # Simulate extracted whistle durations (in seconds)
        tdur_values = np.array([0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3])

        # Generate histogram
        yvalue, nbins, nth, thval = calc_hist(
            tdur_values, BinRes=0.05, prob=True, Ythreshold=50.0
        )

        # Assertions
        assert len(yvalue) == len(nbins)
        assert np.isclose(np.sum(yvalue), 100.0)  # Probability sums to 100%
        assert nth is not None  # Threshold was found
        assert thval >= 50.0

    def test_file_discovery_and_processing(self):
        """Test discovering files and processing them in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON files
            for i in range(3):
                json_data = [
                    {
                        "class": "w",
                        "confidence": 0.90 + i * 0.02,
                        "bbox": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
                    }
                ]
                json_path = os.path.join(tmpdir, f"image_{i:02d}.json")
                with open(json_path, "w") as f:
                    json.dump(json_data, f)

            # Discover files
            json_files = files_list_creator(tmpdir, filesList_extension=".json")
            json_files.sort()

            # Process each file
            all_whistles = []
            for json_file in json_files:
                with open(json_file, "r") as f:
                    data = json.load(f)
                whistles = [item for item in data if item.get("class") == "w"]
                all_whistles.extend(whistles)

            assert len(json_files) == 3
            assert len(all_whistles) == 3

    def test_multiple_formats_and_conversions(self):
        """Test data conversion between JSON, dict, and DataFrame."""
        # Start with JSON string
        json_str = """
        [
            {"class": "w", "confidence": 0.95, "bbox": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}},
            {"class": "w", "confidence": 0.88, "bbox": {"xmin": 200, "ymin": 200, "xmax": 300, "ymax": 300}}
        ]
        """

        # Step 1: JSON → Python dict
        data = json.loads(json_str)
        assert isinstance(data, list)

        # Step 2: Dict → DataFrame
        df = pd.DataFrame(data)
        assert len(df) == 2

        # Step 3: Modify / analyze
        df["width"] = df["bbox"].apply(lambda x: x["xmax"] - x["xmin"])
        assert df["width"].iloc[0] == 100

        # Step 4: Back to JSON
        json_output = df.to_json(orient="records")
        data_restored = json.loads(json_output)
        assert len(data_restored) == 2

    def test_confidence_threshold_filtering(self):
        """Test filtering by confidence threshold across pipeline."""
        detections = [
            {"class": "w", "confidence": 0.95},
            {"class": "w", "confidence": 0.75},
            {"class": "w", "confidence": 0.65},
            {"class": "w", "confidence": 0.55},
        ]

        # Filter by threshold
        threshold = 0.70
        filtered = [d for d in detections if d["confidence"] >= threshold]

        assert len(filtered) == 2
        assert all(d["confidence"] >= threshold for d in filtered)
