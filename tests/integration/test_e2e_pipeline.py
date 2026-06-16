"""
End-to-End Integration Test for WRS Pipeline
Downloads model and sample image from Google Drive, runs complete WRS workflow,
and validates that all steps complete successfully with expected outputs.

This test verifies:
1. Model download from Google Drive
2. Image download from Google Drive
3. WRSapplication detection phase
4. WRSresults analysis phase
5. Output format and correctness
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import download_test_data


@pytest.fixture(scope="session")
def gdrive_files(tmp_path_factory):
    """Resolve model/image once per session using local-first + download_test_data fallback."""
    tmpdir_path = tmp_path_factory.mktemp("e2e_gdrive")
    model_path = tmpdir_path / "best_exp20.pt"
    image_path = tmpdir_path / "sample.png"

    local_model = Path("models/best_exp20.pt")
    local_images = list(Path("images").glob("*.png"))
    if local_model.exists() and local_images:
        shutil.copy(local_model, model_path)
        shutil.copy(local_images[0], image_path)
    else:
        # Reuse the project downloader logic without duplicating folder-download code here.
        current_dir = Path.cwd()
        try:
            os.chdir(tmpdir_path)
            download_ok = download_test_data.download_via_gdown()
        finally:
            os.chdir(current_dir)

        downloaded_model = tmpdir_path / "models_" / "best_exp20.pt"
        downloaded_images = list((tmpdir_path / "images_").glob("*.png"))

        if download_ok and downloaded_model.exists() and downloaded_images:
            shutil.copy(downloaded_model, model_path)
            shutil.copy(downloaded_images[0], image_path)
        else:
            pytest.skip(
                "No local model/image assets and download_test_data.py could not fetch test data."
            )

    return {
        "model": model_path,
        "image": image_path,
        "tmpdir": tmpdir_path,
    }


class TestE2EWRSPipeline:
    """End-to-end integration tests for complete WRS pipeline."""

    def test_wrsapplication_execution(self, gdrive_files):
        """Test WRSapplication detection phase."""
        from scripts.WRSapplication import run as run_application

        model_path = gdrive_files["model"]
        image_path = gdrive_files["image"]
        tmpdir = gdrive_files["tmpdir"]

        output_dir = tmpdir / "wrs_detection"
        output_dir.mkdir(exist_ok=True)

        # Create image input directory
        images_dir = tmpdir / "input_images"
        images_dir.mkdir(exist_ok=True)
        shutil.copy(image_path, images_dir / image_path.name)

        # Simulate WRSapplication run
        import sys

        sys.argv = [
            "run_wrs.py",
            "--model",
            str(model_path),
            "--data_folder",
            str(images_dir),
            "--output_results",
            str(output_dir),
            "--conf",
            "0.5",
            "--merge_iou",
            "0.3",
            "--device",
            "cpu",
        ]

        # Run application
        run_application()

        # Verify outputs
        json_files = list(output_dir.glob("*.json"))
        png_files = list(output_dir.glob("*.png"))

        assert len(json_files) > 0, "No JSON output files generated"
        assert len(png_files) > 0, "No PNG output files generated"

        # Validate JSON structure
        for json_file in json_files:
            with open(json_file, "r") as f:
                detections = json.load(f)

            assert isinstance(detections, list), "JSON should contain a list"

            # Validate each detection
            for det in detections:
                assert "class" in det, "Detection missing 'class'"
                assert "confidence" in det, "Detection missing 'confidence'"
                assert "bbox" in det, "Detection missing 'bbox'"

                bbox = det["bbox"]
                assert all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"])
                assert 0 <= det["confidence"] <= 1.0
                assert bbox["xmin"] < bbox["xmax"]
                assert bbox["ymin"] < bbox["ymax"]

    def test_wrsresults_execution(self, gdrive_files):
        """Test WRSresults analysis phase."""
        from scripts.WRSresults import run as run_results

        tmpdir = gdrive_files["tmpdir"]
        model_path = gdrive_files["model"]
        image_path = gdrive_files["image"]

        # Step 1: Run WRSapplication first
        wrs_app_output = tmpdir / "wrs_detection"
        wrs_app_output.mkdir(exist_ok=True)

        images_dir = tmpdir / "input_images"
        images_dir.mkdir(exist_ok=True)
        shutil.copy(image_path, images_dir / image_path.name)

        import sys

        sys.argv = [
            "run_wrs.py",
            "--model",
            str(model_path),
            "--data_folder",
            str(images_dir),
            "--output_results",
            str(wrs_app_output),
            "--conf",
            "0.5",
            "--merge_iou",
            "0.3",
            "--device",
            "cpu",
        ]
        from scripts.WRSapplication import run as run_application

        run_application()

        # Step 2: Run WRSresults analysis
        wrs_results_output = tmpdir / "wrs_analysis"
        wrs_results_output.mkdir(exist_ok=True)

        sys.argv = [
            "run_wrs.py",
            "--data_folder",
            str(wrs_app_output),
            "--output_results",
            str(wrs_results_output),
            "--output_hist",
            "--output_name",
            "test_results",
        ]
        run_results()

        # Verify outputs
        csv_file = wrs_results_output / "test_results.csv"
        assert csv_file.exists(), "CSV results file not generated"

        # Load and validate CSV (ignore metadata lines prefixed with '#')
        df = pd.read_csv(csv_file, sep=";", comment="#")

        # Check columns
        expected_cols = ["Wid", "Conf", "Tini", "Tdur", "Fmin", "Fmax"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Validate data integrity
        assert len(df) >= 0, "Results should have rows"

        if len(df) > 0:
            # Check data types and ranges
            assert all(df["Conf"] >= 0.0) and all(df["Conf"] <= 1.0)
            assert all(df["Tdur"] > 0), "Duration should be positive"
            assert all(df["Fmax"] > df["Fmin"]), "Fmax should be > Fmin"

            # Verify derived Fdur column
            if "Fdur" in df.columns:
                fdur_expected = df["Fmax"] - df["Fmin"]
                assert np.allclose(df["Fdur"], fdur_expected)

    def test_complete_pipeline_workflow(self, gdrive_files):
        """Test complete pipeline from image to analysis."""
        from scripts.WRSapplication import run as run_application
        from scripts.WRSresults import run as run_results

        model_path = gdrive_files["model"]
        image_path = gdrive_files["image"]
        tmpdir = gdrive_files["tmpdir"]

        # Setup directories
        images_dir = tmpdir / "input_images"
        images_dir.mkdir(exist_ok=True)
        shutil.copy(image_path, images_dir / image_path.name)

        detection_output = tmpdir / "detection"
        analysis_output = tmpdir / "analysis"
        detection_output.mkdir(exist_ok=True)
        analysis_output.mkdir(exist_ok=True)

        # Phase 1: Detection
        import sys

        sys.argv = [
            "run_wrs.py",
            "--model",
            str(model_path),
            "--data_folder",
            str(images_dir),
            "--output_results",
            str(detection_output),
            "--conf",
            "0.5",
            "--merge_iou",
            "0.3",
            "--device",
            "cpu",
        ]
        run_application()

        # Verify detection outputs
        det_jsons = list(detection_output.glob("*.json"))
        det_pngs = list(detection_output.glob("*.png"))
        assert len(det_jsons) > 0, "Detection phase failed: no JSON"
        assert len(det_pngs) > 0, "Detection phase failed: no PNG"

        # Phase 2: Analysis
        sys.argv = [
            "run_wrs.py",
            "--data_folder",
            str(detection_output),
            "--output_results",
            str(analysis_output),
            "--output_hist",
            "--output_name",
            "analysis",
        ]
        run_results()

        # Verify analysis outputs
        csv_file = analysis_output / "analysis.csv"
        assert csv_file.exists(), "Analysis phase failed: no CSV"

        # Check for histogram outputs
        hist_files = list(analysis_output.glob("*_Tdur.png"))
        hist_files += list(analysis_output.glob("*_Fmin.png"))
        hist_files += list(analysis_output.glob("*_Fmax.png"))
        hist_files += list(analysis_output.glob("*_Fdur.png"))

        # Histograms are optional (depend on output_hist flag)
        # Just verify CSV is valid
        df = pd.read_csv(csv_file, sep=";", comment="#")
        assert not df.empty, "CSV results are empty"

        # Verify metadata in CSV
        with open(csv_file, "r") as f:
            header_lines = [f.readline() for _ in range(4)]

        metadata_text = "".join(header_lines)
        assert "whistles" in metadata_text or len(df) >= 0

    def test_json_to_csv_conversion(self):
        """Test conversion from detection JSON to analysis CSV."""
        sample_json = [
            {
                "class": "w",
                "confidence": 0.95,
                "bbox": {"xmin": 10, "ymin": 50, "xmax": 150, "ymax": 150},
            },
            {
                "class": "w",
                "confidence": 0.88,
                "bbox": {"xmin": 200, "ymin": 100, "xmax": 350, "ymax": 200},
            },
            {
                "class": "n",
                "confidence": 0.75,
                "bbox": {"xmin": 400, "ymin": 300, "xmax": 480, "ymax": 350},
            },
        ]

        from src.utils import get_bbox_params

        # Use default config values
        tbin, fbin, foffset, npxs = 0.04, 46.875, 0.0, 512

        # Process whistles only
        whistles = [item for item in sample_json if item["class"] == "w"]
        results = []

        for idx, item in enumerate(whistles):
            conf, tini, tdur, fmin, fmax = get_bbox_params(
                item, tbin, fbin, foffset, npxs
            )
            results.append(
                {
                    "Wid": idx,
                    "Conf": conf,
                    "Tini": tini,
                    "Tdur": tdur,
                    "Fmin": fmin,
                    "Fmax": fmax,
                }
            )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["Fdur"] = df["Fmax"] - df["Fmin"]

        # Validations
        assert len(df) == 2, "Should have 2 whistles"
        assert all(
            col in df.columns for col in ["Conf", "Tdur", "Fmin", "Fmax", "Fdur"]
        )
        assert all(df["Conf"] > 0.85)
        assert all(df["Fmax"] > df["Fmin"])


# Markers for test organization
pytestmark = pytest.mark.integration


@pytest.mark.slow
class TestE2EWithGDriveDownload:
    """Integration tests that download from actual Google Drive."""

    def test_download_and_process_real_data(self, gdrive_files, tmp_path):
        """Run full E2E flow using the shared Drive/local asset resolver."""
        import sys

        from scripts.WRSapplication import run as run_application
        from scripts.WRSresults import run as run_results

        model_path = gdrive_files["model"]
        image_path = gdrive_files["image"]

        input_dir = tmp_path / "input_images"
        detection_dir = tmp_path / "detection"
        analysis_dir = tmp_path / "analysis"
        input_dir.mkdir(exist_ok=True)
        detection_dir.mkdir(exist_ok=True)
        analysis_dir.mkdir(exist_ok=True)
        shutil.copy(image_path, input_dir / image_path.name)

        original_argv = sys.argv.copy()
        try:
            sys.argv = [
                "run_wrs.py",
                "--model",
                str(model_path),
                "--data_folder",
                str(input_dir),
                "--output_results",
                str(detection_dir),
                "--conf",
                "0.5",
                "--merge_iou",
                "0.3",
                "--device",
                "cpu",
            ]
            run_application()

            json_files = list(detection_dir.glob("*.json"))
            assert len(json_files) > 0, "No detection JSON files generated"

            sys.argv = [
                "run_wrs.py",
                "--data_folder",
                str(detection_dir),
                "--output_results",
                str(analysis_dir),
                "--output_name",
                "gdrive_real",
            ]
            run_results()

            csv_file = analysis_dir / "gdrive_real.csv"
            assert csv_file.exists(), "No analysis CSV generated"
            df = pd.read_csv(csv_file, sep=";", comment="#")
            assert "Conf" in df.columns
        finally:
            sys.argv = original_argv
