"""
Script to test a YOLO model using the ultralytics library and Whistles Recognition System model (v0.1.5)

@author: cento123

"""

import argparse
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from whistles_recognition_system.functions import test_model, paint_results, save_jsons

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Set up argument parser
"""
argparser = argparse.ArgumentParser(description="Test YOLO model on a dataset")
argparser.add_argument(
    "--model",
    type=str,
    default="../models/best_exp20.pt",
    help="Path to the YOLO model file",
)
argparser.add_argument(
    "--data", type=str, default="../images/test", help="Path to the dataset for testing"
)
argparser.add_argument(
    "--conf", type=float, default=0.25, help="Confidence threshold for detection"
)
argparser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
argparser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for testing"
)
argparser.add_argument(
    "--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda)"
)
argparser.add_argument(
    "--output_results",
    type=str,
    default="../results",
    help="folder to store results with painted detections",
)


if __name__ == "__main__":
    args = argparser.parse_args()
    model_path = args.model
    data_path = args.data
    conf = args.conf
    iou = args.iou
    batch_size = args.batch_size
    device = args.device
    output_results = args.output_results

    results = test_model(model_path, data_path, conf, iou, batch_size, device)
    paint_results(results, save_path=output_results)
    save_jsons(results, save_path=output_results)
