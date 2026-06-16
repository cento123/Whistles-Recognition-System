"""
Quick validation script to check if test fixes are working.
Run from project root: python validate_tests.py
"""

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and report results."""
    logger.info("%s", "=" * 60)
    logger.info("🧪 %s", description)
    logger.info("%s", "=" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    """Run validation checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checks = [
        (
            "pytest tests/unit/test_utils.py::TestCalcHist::test_calc_hist_uniform_distribution -v",
            "Test 1: Histogram bins fix",
        ),
        (
            "pytest tests/unit/test_wrsapplication.py::TestWRSApplication::test_model_loading_mock -v",
            "Test 2: Model loading mock fix",
        ),
        (
            "pytest tests/unit/test_wrsapplication.py::TestWRSApplication::test_merge_iou_threshold_impact -v",
            "Test 3: IoU threshold impact fix",
        ),
        (
            "pytest tests/validation/test_results.py::TestResultsValidation::test_merged_boxes_consistency -v",
            "Test 4: Box merging consistency fix",
        ),
        (
            "pytest tests/validation/test_results.py::TestResultsValidation::test_csv_metadata_preservation -v",
            "Test 5: CSV metadata preservation fix",
        ),
        (
            "pytest tests/validation/test_results.py::TestResultsValidation::test_iou_threshold_effect_on_count -v",
            "Test 6: IoU threshold effect fix",
        ),
    ]

    passed = 0
    failed = 0

    for cmd, desc in checks:
        if run_command(cmd, desc):
            logger.info("✅ PASSED")
            passed += 1
        else:
            logger.error("❌ FAILED")
            failed += 1

    logger.info("%s", "=" * 60)
    logger.info(
        "📊 Summary: %s passed, %s failed out of %s tests", passed, failed, len(checks)
    )
    logger.info("%s", "=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
