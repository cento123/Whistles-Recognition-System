#!/usr/bin/env python3
"""
Quick launcher for WRS End-to-End Tests
Runs complete pipeline: Download → Test → Validate
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Execute command and report results."""
    logger.info("%s", "=" * 70)
    logger.info("▶ %s", description)
    logger.info("%s", "=" * 70)
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def check_files():
    """Check if required files exist."""
    model = Path("models/best_exp20.pt").exists()
    images = len(list(Path("images").glob("*.png"))) > 0

    logger.info("Model:  %s", "✅" if model else "❌")
    logger.info("Images: %s", "✅" if images else "❌")

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

    logger.info("%s", "=" * 70)
    logger.info("🎵 WRS End-to-End Test Launcher")
    logger.info("%s", "=" * 70)

    # Download if needed
    if args.download or not check_files():
        logger.info("📥 Downloading test data...")
        if run_command("python download_test_data.py", "Download from Google Drive"):
            logger.info("✅ Download complete")
        else:
            logger.warning("⚠️ Download failed or skipped")

    # Verify files
    logger.info("📋 Checking files...")
    if not check_files():
        logger.error("❌ Required files missing")
        logger.error("Run: python download_test_data.py")
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

    logger.info("🧪 Running: %s", args.test)
    success = run_command(cmd, "Running WRS E2E Tests")

    logger.info("%s", "=" * 70)
    if success:
        logger.info("✅ All tests passed!")
        logger.info("%s", "=" * 70)
        return 0
    else:
        logger.error("❌ Some tests failed")
        logger.info("%s", "=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
