"""
Run the Whistles Recognition System on all images in a specified folder, including its subfolders.
Processes each image with a YOLO model, saves annotated images, and exports JSON results.
Supports filtering by filename substring and image extension, with configurable confidence and IoU thresholds.

# Created on Thu Mar 12 2026 10:25:02 UTC
@author: ddietor & cento123
"""

import argparse
import logging
import os
import shutil

from src.utils import files_list_creator, paint_results, save_jsons, test_model

SCRIPT_NAME = "WRSapplication"


def run():
    """Main function to run the Whistles Recognition System."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up argument parser
    argparser = argparse.ArgumentParser(
        description=(
            "Script to run Whistles-Recognition-System on images and save the results.\n"
            "It generates visualizations and JSON files for each processed image.\n"
            "Example usage:\n"
            "  python run_wrs.py application --model ./models/best_exp20.pt --data_folder ./data --output_results ./results"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="./models/best_exp20.pt",
        help="Path to the YOLO model file",
    )
    argparser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold for detection (default: 0.50)",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for testing"
    )
    argparser.add_argument(
        "--merge_iou",
        type=float,
        default=0.3,
        help="IoU threshold to merge overlapping detections (default: 0.3)",
    )
    argparser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (cpu or cuda)",
    )
    argparser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Folder containing the images to process (mandatory).",
    )
    argparser.add_argument(
        "--output_results",
        type=str,
        default=".",
        help="Folder to store analysis results (default: current folder).",
    )
    argparser.add_argument(
        "--image_extension",
        type=str,
        default=".png",
        help="File extension of images to process: .png, .jpg, etc. (default: .png)",
    )
    argparser.add_argument(
        "--filename_contains",
        type=str,
        help="Optional substring that filenames must contain.",
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    # Parse arguments
    args = argparser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    else:
        logger.setLevel(logging.INFO)

    # Remove the output_results folder if it already exists to avoid mixing old and new results
    if os.path.exists(args.output_results):
        shutil.rmtree(args.output_results)

    # Create the output_results folder if not exists
    os.makedirs(args.output_results, exist_ok=True)

    images = files_list_creator(
        args.data_folder,
        filesList_extension=args.image_extension,
        filesList_contains=args.filename_contains,
    )
    logger.info(f"There are {len(images)} images to analyze")

    for indx_png, image in enumerate(images, start=1):
        logger.info(f"Processing ({indx_png}/{len(images)}): {os.path.basename(image)}")
        results = test_model(
            args.model, image, args.conf, args.merge_iou, args.batch_size, args.device
        )
        if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
            logger.info(
                f"Found {len(results[0].boxes)} whistle(s) in {os.path.basename(image)}"
            )
            paint_results(
                results, save_path=args.output_results, merge_iou=args.merge_iou
            )
            save_jsons(results, save_path=args.output_results, merge_iou=args.merge_iou)

    logger.info(f"...{SCRIPT_NAME} finalize!")


if __name__ == "__main__":
    run()
