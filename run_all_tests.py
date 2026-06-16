#!/usr/bin/env python3
"""
Re-run all tests to verify fixes are working.
Execute from project root: python run_all_tests.py
"""

import os
import subprocess
import sys


def main():
    """Run all tests with fresh Python environment."""

    print("\n" + "=" * 70)
    print("🧪 RUNNING ALL TESTS - Fresh Execution")
    print("=" * 70 + "\n")

    # Run all tests
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"], cwd=os.getcwd()
    )

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed. See above for details.")
    print("=" * 70 + "\n")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
