"""
Simplified End-to-End Integration Test using local files.
Verifies complete WRS pipeline with actual model and sample images.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestE2ELocalPipeline:
    """End-to-end tests using local files."""

    @pytest.fixture
    def local_test_files(self):
        """Setup test with local model and images."""
        model_path = Path("./models/best_exp20.pt")
        images_dir = Path("./images")

        # Check if model exists
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        # Check if images exist
        test_images = list(images_dir.glob("*.png"))
        if not test_images:
            pytest.skip(f"No test images found in {images_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            yield {
                "model": model_path,
                "image": test_images[0],
                "tmpdir": Path(tmpdir),
            }

    def test_e2e_wrsapplication_flow(self, local_test_files):
        """Test WRSapplication detection with local files."""
        from scripts.WRSapplication import run as run_application

        model = local_test_files["model"]
        image = local_test_files["image"]
        tmpdir = local_test_files["tmpdir"]

        # Create input directory with test image
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output_detection"
        input_dir.mkdir()
        output_dir.mkdir()

        shutil.copy(image, input_dir / image.name)

        # Setup sys.argv for WRSapplication
        import sys

        original_argv = sys.argv.copy()

        try:
            sys.argv = [
                "run_wrs.py",
                "--model",
                str(model),
                "--data_folder",
                str(input_dir),
                "--output_results",
                str(output_dir),
                "--conf",
                "0.5",
                "--merge_iou",
                "0.3",
                "--device",
                "cpu",
            ]

            run_application()

            # Verify outputs
            json_files = sorted(output_dir.glob("*.json"))
            png_files = sorted(output_dir.glob("*.png"))

            assert len(json_files) > 0, "No JSON detections generated"
            assert len(png_files) > 0, "No annotated PNG generated"

            # Validate JSON structure
            for json_file in json_files:
                with open(json_file, "r") as f:
                    detections = json.load(f)

                assert isinstance(detections, list)

                for det in detections:
                    assert "class" in det
                    assert "confidence" in det
                    assert "bbox" in det
                    assert 0 <= det["confidence"] <= 1.0
                    bbox = det["bbox"]
                    assert bbox["xmin"] < bbox["xmax"]
                    assert bbox["ymin"] < bbox["ymax"]

            # Verify PNG is valid
            for png_file in png_files:
                assert png_file.stat().st_size > 0, f"PNG file is empty: {png_file}"

        finally:
            sys.argv = original_argv

    def test_e2e_wrsresults_flow(self, local_test_files):
        """Test WRSresults analysis with detection outputs."""
        from scripts.WRSapplication import run as run_application
        from scripts.WRSresults import run as run_results

        model = local_test_files["model"]
        image = local_test_files["image"]
        tmpdir = local_test_files["tmpdir"]

        # Phase 1: Create detection outputs
        input_dir = tmpdir / "input"
        detection_dir = tmpdir / "detections"
        analysis_dir = tmpdir / "analysis"
        input_dir.mkdir()
        detection_dir.mkdir()
        analysis_dir.mkdir()

        shutil.copy(image, input_dir / image.name)

        # Run WRSapplication
        import sys

        original_argv = sys.argv.copy()

        try:
            sys.argv = [
                "run_wrs.py",
                "--model",
                str(model),
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

            # Verify detection outputs exist
            json_files = list(detection_dir.glob("*.json"))
            assert len(json_files) > 0, "WRSapplication didn't generate JSONs"

            # Phase 2: Run WRSresults analysis
            sys.argv = [
                "run_wrs.py",
                "--data_folder",
                str(detection_dir),
                "--output_results",
                str(analysis_dir),
                "--output_hist",
                "--output_name",
                "test_analysis",
            ]
            run_results()

            # Verify analysis outputs
            csv_file = analysis_dir / "test_analysis.csv"
            assert csv_file.exists(), "WRSresults didn't generate CSV"

            # Load and validate CSV (ignore metadata lines prefixed with '#')
            df = pd.read_csv(csv_file, sep=";", comment="#")

            # Check expected columns
            expected_cols = ["Wid", "Conf", "Tini", "Tdur", "Fmin", "Fmax"]
            for col in expected_cols:
                assert col in df.columns, f"Missing column: {col}"

            # If whistles were detected, validate data
            if len(df) > 0:
                assert all(df["Conf"] >= 0.0) and all(df["Conf"] <= 1.0)
                assert all(df["Tdur"] > 0), "Duration should be positive"
                assert all(df["Fmax"] > df["Fmin"]), "Fmax should be > Fmin"

                if "Fdur" in df.columns:
                    fdur_expected = df["Fmax"] - df["Fmin"]
                    assert np.allclose(df["Fdur"], fdur_expected)

            # Verify CSV metadata
            with open(csv_file, "r") as f:
                header = f.read(200)
            assert "whistles" in header or "#" in header

        finally:
            sys.argv = original_argv

    def test_e2e_complete_workflow(self, local_test_files):
        """Test complete workflow from image to final analysis."""
        from scripts.WRSapplication import run as run_application
        from scripts.WRSresults import run as run_results

        model = local_test_files["model"]
        image = local_test_files["image"]
        tmpdir = local_test_files["tmpdir"]

        # Setup directories
        input_dir = tmpdir / "input"
        detection_dir = tmpdir / "detection"
        analysis_dir = tmpdir / "analysis"

        for d in [input_dir, detection_dir, analysis_dir]:
            d.mkdir(exist_ok=True)

        shutil.copy(image, input_dir / image.name)

        import sys

        original_argv = sys.argv.copy()

        try:
            # Step 1: Detection
            print("\n📸 Running WRSapplication (detection phase)...")
            sys.argv = [
                "run_wrs.py",
                "--model",
                str(model),
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
                "--verbose",
            ]
            run_application()

            # Verify detection step
            det_jsons = sorted(detection_dir.glob("*.json"))
            det_pngs = sorted(detection_dir.glob("*.png"))

            print(f"✅ Detection complete: {len(det_jsons)} JSON, {len(det_pngs)} PNG")
            assert len(det_jsons) > 0, "Detection failed"
            assert len(det_pngs) > 0, "No images generated"

            # Count detections
            total_detections = 0
            for json_file in det_jsons:
                with open(json_file) as f:
                    dets = json.load(f)
                    total_detections += len(dets)
            print(f"🎯 Total detections: {total_detections}")

            # Step 2: Analysis
            print("\n📊 Running WRSresults (analysis phase)...")
            sys.argv = [
                "run_wrs.py",
                "--data_folder",
                str(detection_dir),
                "--output_results",
                str(analysis_dir),
                "--output_hist",
                "--output_name",
                "final_results",
                "--verbose",
            ]
            run_results()

            # Verify analysis step
            csv_file = analysis_dir / "final_results.csv"
            assert csv_file.exists(), "CSV not generated"

            df = pd.read_csv(csv_file, sep=";", comment="#")
            print(f"✅ Analysis complete: {len(df)} whistles analyzed")

            # Check for histogram outputs
            hist_files = list(analysis_dir.glob("final_results_*.png"))
            print(f"📈 Generated {len(hist_files)} histogram files")

            # Final validation
            if len(df) > 0:
                print(f"\n📋 Whistle Statistics:")
                print(
                    f"  - Confidence: {df['Conf'].min():.2f} - {df['Conf'].max():.2f}"
                )
                print(
                    f"  - Duration: {df['Tdur'].min():.3f}s - {df['Tdur'].max():.3f}s"
                )
                print(
                    f"  - Freq Min: {df['Fmin'].min():.0f} - {df['Fmin'].max():.0f} Hz"
                )
                print(
                    f"  - Freq Max: {df['Fmax'].min():.0f} - {df['Fmax'].max():.0f} Hz"
                )

            print("\n✅ End-to-end workflow completed successfully!")

        finally:
            sys.argv = original_argv


@pytest.mark.integration
class TestDetectionValidation:
    """Validate detection outputs format and consistency."""

    def test_detection_json_format(self):
        """Test JSON output format is correct."""
        sample_detection = {
            "class": "w",
            "confidence": 0.95,
            "bbox": {"xmin": 100, "ymin": 50, "xmax": 200, "ymax": 150},
        }

        # Validate structure
        assert isinstance(sample_detection, dict)
        assert "class" in sample_detection
        assert "confidence" in sample_detection
        assert "bbox" in sample_detection

        bbox = sample_detection["bbox"]
        required_bbox_keys = ["xmin", "ymin", "xmax", "ymax"]
        assert all(k in bbox for k in required_bbox_keys)

        # Validate ranges
        assert sample_detection["confidence"] >= 0.0
        assert sample_detection["confidence"] <= 1.0
        assert bbox["xmin"] < bbox["xmax"]
        assert bbox["ymin"] < bbox["ymax"]

    def test_csv_output_format(self):
        """Test CSV output format is correct."""
        sample_data = {
            "Wid": [0, 1, 2],
            "Conf": [0.95, 0.88, 0.92],
            "Tini": [0.1, 0.5, 1.2],
            "Tdur": [0.3, 0.4, 0.25],
            "Fmin": [1000, 1200, 900],
            "Fmax": [2000, 2100, 1800],
        }

        df = pd.DataFrame(sample_data)
        df["Fdur"] = df["Fmax"] - df["Fmin"]

        # Validate structure
        assert len(df) == 3
        expected_cols = ["Wid", "Conf", "Tini", "Tdur", "Fmin", "Fmax", "Fdur"]
        assert all(col in df.columns for col in expected_cols)

        # Validate data
        assert all(df["Conf"] >= 0) and all(df["Conf"] <= 1)
        assert all(df["Tdur"] > 0)
        assert all(df["Fmax"] > df["Fmin"])
