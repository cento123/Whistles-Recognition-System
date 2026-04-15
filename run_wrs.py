"""
Launcher script for WRS - Run from the project root.
Usage:
    python run_wrs.py application --data_folder ./images/test --output_results ./WRSapplication_results
    python run_wrs.py results --data_folder ./WRSapplication_results --output_results ./WRSresults_results
"""

import sys

if len(sys.argv) < 2:
    print("Usage:")
    print("  python run_wrs.py application [args...]")
    print("  python run_wrs.py results [args...]")
    sys.exit(1)

command = sys.argv[1]
sys.argv = sys.argv[1:]  # Remove 'application' or 'results' from args

if command == "application":
    from scripts.WRSapplication import run as run_application

    run_application()
elif command == "results":
    from scripts.WRSresults import run as run_results

    run_results()
else:
    print(f"Unknown command: {command}")
    print("Use 'application' or 'results'")
    sys.exit(1)
