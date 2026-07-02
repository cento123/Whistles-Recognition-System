"""
Launcher script for WRS - Run from the project root.
Usage:
    python run_wrs.py application --data_folder ./images/test --output_results ./WRSapplication_results
    python run_wrs.py results --data_folder ./WRSapplication_results --output_results ./WRSresults_results
"""

import logging
import sys
### Bad comment to remove
logger = logging.getLogger(__name__)


def main():
    """Dispatch to the requested WRS workflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        logger.error("Usage:")
        logger.error("  python run_wrs.py application [args...]")
        logger.error("  python run_wrs.py results [args...]")
        return 1

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove 'application' or 'results' from args

    if command == "application":
        from scripts.WRSapplication import run as run_application

        run_application()
    elif command == "results":
        from scripts.WRSresults import run as run_results

        run_results()
    else:
        logger.error("Unknown command: %s", command)
        logger.error("Use 'application' or 'results'")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
