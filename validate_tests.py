"""
Quick validation script to check if test fixes are working.
Run from project root: python validate_tests.py
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    """Run validation checks."""
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
            print(f"✅ PASSED")
            passed += 1
        else:
            print(f"❌ FAILED")
            failed += 1

    print(f"\n{'='*60}")
    print(f"📊 Summary: {passed} passed, {failed} failed out of {len(checks)} tests")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
