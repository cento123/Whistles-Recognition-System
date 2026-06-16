#!/usr/bin/env python3
"""
Quick launcher for WRS End-to-End Tests
Runs complete pipeline: Download → Test → Validate
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Execute command and report results."""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def check_files():
    """Check if required files exist."""
    model = Path("models_/best_exp20.pt").exists()
    images = len(list(Path("images_").glob("*.png"))) > 0

    print(f"Model:  {'✅' if model else '❌'}")
    print(f"Images: {'✅' if images else '❌'}")

    return model and images


def main():
    """Main launcher."""
    import argparse

    parser = argparse.ArgumentParser(description="WRS E2E Test Launcher")
    parser.add_argument(
        "--download", action="store_true", help="Download test data first"
    )
    parser.add_argument(
        "--test",
        default="complete",
        help="Test to run: complete, detection, analysis, validation",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎵 WRS End-to-End Test Launcher")
    print("=" * 70)

    # Download if needed
    if args.download or not check_files():
        print("\n📥 Downloading test data...")
        if run_command("python download_test_data.py", "Download from Google Drive"):
            print("✅ Download complete")
        else:
            print("⚠️ Download failed or skipped")

    # Verify files
    print("\n📋 Checking files...")
    if not check_files():
        print("❌ Required files missing")
        print("\nRun: python download_test_data.py")
        return 1

    # Run tests
    test_map = {
        "complete": "tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_complete_workflow",
        "detection": "tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_wrsapplication_flow",
        "analysis": "tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_wrsresults_flow",
        "validation": "tests/integration/test_e2e_local.py::TestDetectionValidation",
        "all": "tests/integration/test_e2e_local.py",
    }

    test_path = test_map.get(args.test, test_map["complete"])

    verbose = "-vv -s" if args.verbose else "-v -s"
    cmd = f"pytest {test_path} {verbose} --tb=short"

    print(f"\n🧪 Running: {args.test}")
    success = run_command(cmd, "Running WRS E2E Tests")

    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed!")
        print("=" * 70)
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
