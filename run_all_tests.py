#!/usr/bin/env python3
"""
Re-run all tests to verify fixes are working.
Execute from project root: python run_all_tests.py
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def main():
    """Run all tests with fresh Python environment."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    separator = "=" * 70
    logger.info("%s", separator)
    logger.info("🧪 RUNNING ALL TESTS - Fresh Execution")
    logger.info("%s", separator)

    # Run all tests
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"], cwd=os.getcwd()
    )

    logger.info("%s", separator)
    if result.returncode == 0:
        logger.info("✅ ALL TESTS PASSED!")
    else:
        logger.error("❌ Some tests failed. See above for details.")
    logger.info("%s", separator)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
