"""
Script to test a YOLO model using the ultralytics library and Whistles Recognition System model (v0.1.5)
"""

import argparse
import json
import logging
import os

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
loger = logging.getLogger(__name__)

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


def test_model(
    model_path: str,
    data_path: str,
    conf: float = 0.50,
    iou: float = 0.25,
    batch_size: int = 16,
    device: str = "cpu",
) -> DetMetrics:
    """
    Test the YOLO model on the specified dataset.
    Args:
        model_path (str): Path to the YOLO model file.
        data_path (str): Path to the dataset for testing.
        conf (float): Confidence threshold for detection.
        iou (float): IoU threshold for NMS.
        batch_size (int): Batch size for testing.
        device (str): Device to run the model on (cpu or cuda).
    Returns:
        results: list of Detection results from the model.
    """

    loger.info(
        f"Testing model: {model_path} on data: {data_path} with conf: {conf}, iou: {iou}, batch_size: {batch_size}, device: {device}"
    )
    # Load the YOLO model
    model = YOLO(model_path)

    # Run the test
    results = model.predict(
        source=data_path,
        conf=conf,
        iou=iou,
        batch=batch_size,
        device=device,
        save=False,
    )

    return results


def paint_results(results: DetMetrics, save_path: str = "../results") -> None:
    """
    Paint the detection results on the images and save them.
    Args:
        results: list of Detection results from the model.
        save_path (str): Directory to save the painted images.
    """

    loger.info(f"Painting results and saving to: {save_path}")

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Draw rectangles and labels on the images
    for result in results:
        image_path = result.path
        img = cv2.imread(image_path)

        for bboxes in result.boxes:
            for box in bboxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()

                # Draw rectangle
                img = cv2.rectangle(
                    img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
                )

                # Put label
                label = f"{result.names[int(cls)]}: {conf:.2f}"
                img = cv2.putText(
                    img,
                    label,
                    (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        base_name = os.path.basename(image_path)
        output_path = os.path.join(save_path, base_name)
        cv2.imwrite(output_path, img)


def save_jsons(results: DetMetrics, save_path: str = "../results") -> None:
    """
    Save the detection results to a json files.
    Args:
        results: list of Detection results from the model.
        save_path (str): Path to save the json results files.
    """

    loger.info(f"Saving results to json files in: {save_path}")

    for result in results:
        json_path = os.path.join(
            save_path, os.path.splitext(os.path.basename(result.path))[0] + ".json"
        )
        json_result = []
        with open(json_path, "w") as f:
            for bboxes in result.boxes:
                for box in bboxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    json_result.append(
                        {
                            "class": result.names[int(cls)],
                            "confidence": np.round(float(conf), 2),
                            "bbox": {
                                "xmin": xmin,
                                "ymin": ymin,
                                "xmax": xmax,
                                "ymax": ymax,
                            },
                        }
                    )
            json.dump(json_result, f, indent=4)


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
