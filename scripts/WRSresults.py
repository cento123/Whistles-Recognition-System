"""
Analyze Whistles Recognition System JSON outputs.
Extracts Tdur, Fmin, Fmax, Fdur for all whistles, saves a CSV summary, and optionally generates histograms and boxplots.
Supports configurable output folder, file naming, and verbose logging.

# Created on Tue Mar 10 2026 18:56:32 UTC
@author: ddietor
"""

import argparse
import datetime as dt
import json
import logging

#  Imports
# Libraries:
import os

import pandas as pd

# External parameters:
from src.utils import (
    calc_hist,
    files_list_creator,
    get_bbox_params,
    load_config,
    plot_WRSresults,
    save_hist,
)

Tbin, Fbin, Foffset, Npxs = load_config("src/config.yaml")


script_name = "WRSresults"
#  Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Set up argument parser
"""
argparser = argparse.ArgumentParser(
    description="Script to analyze WRS results from the outpu JSON files.\n"
    "It extracts Tdur, Fmin, Fmax, Fdur, and creates a CSV file and some plots.\n"
    "Example usage:\n"
    "  python ./scripts/WRSresults_v1.py --data_folder ./data --output_results ./results --output_name my_analysis --output_hist",
    formatter_class=argparse.RawTextHelpFormatter,
)

argparser.add_argument(
    "--data_folder",
    type=str,
    required=True,
    help="Folder to search for the JSON files (mandatory).",
)
argparser.add_argument(
    "--output_results",
    type=str,
    default=".",
    help="Folder to store analysis results (default: current folder).",
)
argparser.add_argument(
    "--output_hist",
    action="store_true",
    help="If set, saves histograms for each calculated parameter (default: False).",
)
argparser.add_argument(
    "--output_name",
    type=str,
    default=script_name,
    help=f"Base name for output files (CSV, plots). Default: '{script_name}'.",
)
argparser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

# -----------------------------
# Parse arguments
# -----------------------------
args = argparser.parse_args()

# Set logging level based on verbose flag
if args.verbose:
    logger.setLevel(logging.DEBUG)
    logger.debug("Verbose logging enabled.")
else:
    logger.setLevel(logging.INFO)

logger.info(
    f"""
{'='*50}
This is the setup in config.py:
Tbin = {Tbin} s
Fbin = {Fbin} Hz
Foffset = {Foffset} Hz
Npxs = {Npxs}
{'='*50}
"""
)
#  Program execution:
if __name__ == "__main__":
    args = argparser.parse_args()
    data_path = args.data_folder
    output_results = args.output_results
    FileName_output = args.output_name
    PltHist_flag = args.output_hist

    jsons = files_list_creator(data_path, filesList_extension=".json")
    jsons.sort()

    Results_df_array = []
    Results_df_columns = ["FileName", "Wid", "Conf", "Tini", "Tdur", "Fmin", "Fmax"]
    W_id = -1
    for json_file in jsons:
        # Read JSON
        with open(json_file, "r") as f:
            data = json.load(f)
        # Check if empty
        if data:
            data.sort(key=lambda item: item["bbox"]["xmin"])
            for item in data:
                if item.get("class") == "w":
                    W_id += 1
                    Conf, Tini, Tdur, Fmin, Fmax = get_bbox_params(
                        item, Tbin, Fbin, Foffset, Npxs
                    )
                    row_data = [
                        os.path.basename(json_file),
                        W_id,
                        Conf,
                        Tini,
                        Tdur,
                        Fmin,
                        Fmax,
                    ]
                    Results_df_array.append(row_data)
    Results_df = pd.DataFrame(data=Results_df_array, columns=Results_df_columns)
    Results_df["Fdur"] = Results_df["Fmax"] - Results_df["Fmin"]

    # Histograms of data:
    FontSize = 16
    if PltHist_flag:
        not_plot_cols = ["FileName", "Wid", "Conf", "Tini"]
        for col in Results_df.columns:
            if col not in not_plot_cols:
                data2hist = Results_df[col]
                if "F" in col:
                    data2hist = data2hist * 1e-3
                    BinRes = 0.5
                    Xlabel_str = "Frequency [kHz]"
                elif "T" in col:
                    data2hist = data2hist
                    BinRes = 0.05
                    Xlabel_str = "Time [s]"
                else:
                    continue
                Yvalue, Nbins, nTh, thVal = calc_hist(data2hist, BinRes, prob=True)
                title_str = f"{len(Results_df)} whistles\n{col}"
                HistName = FileName_output + f"_{col}"
                save_hist(
                    Yvalue,
                    Nbins,
                    FontSize,
                    title_str,
                    Xlabel_str,
                    output_results,
                    HistName,
                )

    # Output data saved as csv
    logger.info(
        f"{len(jsons)} JSON files analyzed, containing {len(Results_df)} whistles, stored in {FileName_output}"
    )
    metadata = [
        "# ----------------------------------------"
        f"# {len(jsons)} JSON files analyzed, containing {len(Results_df)} whistles",
        f"# Creation: {dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')} by {script_name}"
        f"# Images config: Tbin={Tbin}s, Fbin={Fbin}Hz, Foffset={Foffset}Hz, Npxs={Npxs}",
        "# ----------------------------------------",
    ]
    with open(
        os.path.join(output_results, FileName_output + ".csv"), "w", encoding="utf-8"
    ) as f:
        # Metadata:
        for line in metadata:
            f.write(line + "\n")
        # Data:
        Results_df.to_csv(f, sep=";", index=False)

    # Output data saved as Boxplot:
    plot_WRSresults(Results_df, FontSize, output_results, FileName_output)

    logger.info(f"...{script_name} finalize!")
