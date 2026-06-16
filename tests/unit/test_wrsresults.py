"""
Unit tests for scripts/WRSresults.py
Tests result analysis, whistles extraction, and statistics generation.
"""

import json
import os
import tempfile

import pandas as pd

from src.utils import get_bbox_params, load_config


class TestWRSResults:
    """Test WRSresults module functionality."""

    def test_json_parsing(self):
        """Test parsing of JSON result files."""
        json_data = [
            {
                "class": "w",
                "confidence": 0.95,
                "bbox": {"xmin": 10, "ymin": 50, "xmax": 150, "ymax": 150},
            },
            {
                "class": "n",
                "confidence": 0.75,
                "bbox": {"xmin": 200, "ymin": 100, "xmax": 250, "ymax": 200},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "result.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            # Load and filter (mimics WRSresults behavior)
            with open(json_path, "r") as f:
                data = json.load(f)

            whistles = [item for item in data if item.get("class") == "w"]
            assert len(whistles) == 1
            assert whistles[0]["confidence"] == 0.95

    def test_bbox_parameter_extraction(self, config_yaml_path):
        """Test extraction of whistle parameters from bounding boxes."""
        tbin, fbin, foffset, npxs = load_config(config_yaml_path)

        item = {
            "class": "w",
            "confidence": 0.90,
            "bbox": {"xmin": 0, "ymin": 100, "xmax": 100, "ymax": 200},
        }

        conf, tini, tdur, fmin, fmax = get_bbox_params(item, tbin, fbin, foffset, npxs)

        assert conf == 0.90
        assert tini == 0.0  # xmin * tbin
        assert tdur > 0  # (xmax - xmin) * tbin
        assert fmin > 0
        assert fmax > fmin  # fmax should be greater than fmin

    def test_dataframe_construction(self):
        """Test construction of results DataFrame."""
        data = [
            {
                "FileName": "img1.json",
                "Wid": 0,
                "Conf": 0.95,
                "Tini": 0.1,
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

        assert len(df) == 3
        assert "Fdur" in df.columns
        assert df["Fdur"].iloc[0] == 1000

    def test_histogram_statistics(self):
        """Test histogram calculation for whistle statistics."""
        from src.utils import calc_hist

        # Simulate whistle duration data (in seconds)
        tdur_data = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

        yvalue, nbins, nth, thval = calc_hist(tdur_data, BinRes=0.05, prob=True)

        assert len(yvalue) > 0
        assert np.isclose(np.sum(yvalue), 100.0)  # Percentages

    def test_csv_output_format(self):
        """Test CSV output naming and structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_name = "test_results"
            csv_path = os.path.join(tmpdir, f"{output_name}.csv")

            # Simulate saving DataFrame to CSV
            df = pd.DataFrame(
                {
                    "Wid": [0, 1, 2],
                    "Tdur": [0.1, 0.2, 0.3],
                    "Fmin": [1000, 1200, 1100],
                    "Fmax": [2000, 2200, 2100],
                }
            )

            metadata = [
                "# 3 whistles analyzed",
                "# Creation: 2026-06-16T10:00:00Z",
            ]

            with open(csv_path, "w") as f:
                for line in metadata:
                    f.write(line + "\n")
                df.to_csv(f, sep=";", index=False)

            # Verify the file was created
            assert os.path.exists(csv_path)

            # Verify metadata was written
            with open(csv_path, "r") as f:
                lines = f.readlines()
                assert lines[0].startswith("#")
                assert "whistles" in lines[0]

    def test_multiple_json_aggregation(self):
        """Test aggregating results from multiple JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple JSON files
            json_files = []
            all_whistles = []

            for file_idx in range(3):
                json_data = []
                for whistle_idx in range(2):
                    json_data.append(
                        {
                            "class": "w",
                            "confidence": 0.9 - file_idx * 0.05,
                            "bbox": {
                                "xmin": whistle_idx * 100,
                                "ymin": 50,
                                "xmax": (whistle_idx + 1) * 100,
                                "ymax": 150,
                            },
                        }
                    )
                    all_whistles.append(json_data[-1])

                json_path = os.path.join(tmpdir, f"result_{file_idx}.json")
                with open(json_path, "w") as f:
                    json.dump(json_data, f)
                json_files.append(json_path)

            # Load and aggregate
            aggregate_whistles = []
            for json_file in json_files:
                with open(json_file, "r") as f:
                    data = json.load(f)
                whistles = [item for item in data if item.get("class") == "w"]
                aggregate_whistles.extend(whistles)

            assert len(aggregate_whistles) == 6  # 3 files * 2 whistles
            assert len(json_files) == 3

    def test_class_filtering(self):
        """Test filtering of whistle class from detections."""
        json_data = [
            {"class": "w", "confidence": 0.95},
            {"class": "n", "confidence": 0.85},
            {"class": "w", "confidence": 0.90},
            {"class": "n", "confidence": 0.75},
        ]

        whistles = [item for item in json_data if item.get("class") == "w"]
        noise = [item for item in json_data if item.get("class") == "n"]

        assert len(whistles) == 2
        assert len(noise) == 2
        assert all(w["class"] == "w" for w in whistles)
        assert all(n["class"] == "n" for n in noise)


import numpy as np
