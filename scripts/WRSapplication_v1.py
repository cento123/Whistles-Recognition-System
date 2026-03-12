"""
# Created on Thu Mar 12 2026 10:25:02 UTC

@author: ddietor

"""
# %% Imports
# Libraries:
import argparse
import logging
import os

from whistles_recognition_system.functions import test_model, paint_results, save_jsons, filesListCreator

script_name = "WRSapplication_v1"
# %% Set up logging
# %% Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Set up argument parser
"""
argparser = argparse.ArgumentParser(
    description=(
        "Script to run Whistles-Recognition-System on images and save the results.\n"
        "It generates visualizations and JSON files for each processed image.\n"
        "Requires the 'test.py' file to be in the same folder as this script.\n"
        "Example usage:\n"
        "  python WRSapplication_v1.py --data_folder ./data --output_results ./results --model ../models/best_exp20.pt"
    ),
    formatter_class=argparse.RawTextHelpFormatter
)
argparser.add_argument(
    "--model",
    type=str,
    default="../models/best_exp20.pt",
    help="Path to the YOLO model file",
)
argparser.add_argument(
    "--conf", type=float, default=0.15, help="Confidence threshold for detection (default: 0.15)"
)
argparser.add_argument("--iou", type=float, default=0.40, help="IoU threshold for NMS (default: 0.40)")
argparser.add_argument(
    "--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda)"
)
argparser.add_argument(
    "--data_folder",
    type=str,
    required=True,
    help="Folder containing the images to process (mandatory)."
)
argparser.add_argument(
    "--output_results",
    type=str,
    default=".",
    help="Folder to store analysis results (default: current folder)."
)
argparser.add_argument(
    "--image_extension",
    type=str,
    default=".png",
    help="File extension of images to process: .png, .jpg, etc. (default: .png)"
)
argparser.add_argument(
    "--filename_contains",
    type=str,
    help="Optional substring that filenames must contain."
)
argparser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose logging."
)

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

# %% Program execution:
if __name__ == "__main__":
    # print(f"Executing {script_name}...")
    model_path = args.model
    conf = args.conf
    iou = args.iou
    batch_size = 1
    device = args.device
    data_folder = args.data_folder
    output_results = args.output_results
    filename_contains = args.filename_contains
    image_extension = args.image_extension

    images = filesListCreator(data_folder,filesList_extension=image_extension, filesList_contains=filename_contains)
    print(f"There are {len(images)} images to analyze")
    for indx_png, png in enumerate(images, start=1):
        print(f"Processing ({indx_png+1}/{len(images)}): {os.path.basename(png)}")
        results = test_model(model_path, png, conf, iou, batch_size, device)
        if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
            print(f"Found {len(results[0].boxes)} whistle(s) in {os.path.basename(png)}")
            paint_results(results, save_path=output_results)
            save_jsons(results, save_path=output_results)

    # print(f"...{script_name} finalize!")