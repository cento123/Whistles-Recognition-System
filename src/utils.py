"""
Utility functions for testing the WRS model, visualizing results, and processing detection outputs.
"""

# %% Imports
import json
import logging
import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
import yaml  # type: ignore
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def merge_two_boxes(det1: dict, det2: dict) -> dict:
    """
    The merged box is the union of the two boxes, and the confidence is the max of the two.

    Args:
        det1 (dict): First detection dictionary.
        det2 (dict): Second detection dictionary.

    Returns:
        dict: Merged detection dictionary.
    """
    return {
        "class": det1["class"],
        "confidence": max(det1["confidence"], det2["confidence"]),
        "bbox": {
            "xmin": min(det1["bbox"]["xmin"], det2["bbox"]["xmin"]),
            "ymin": min(det1["bbox"]["ymin"], det2["bbox"]["ymin"]),
            "xmax": max(det1["bbox"]["xmax"], det2["bbox"]["xmax"]),
            "ymax": max(det1["bbox"]["ymax"], det2["bbox"]["ymax"]),
        },
    }


def calc_iou(box1: dict, box2: dict) -> float:
    """
    Calculate IoU between two boxes.

    Args:
        box1 (dict): First box with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        box2 (dict): Second box with keys 'xmin', 'ymin', 'xmax', 'ymax'.

    Returns:
        float: IoU value between 0 and 1.

    """
    x1 = max(box1["xmin"], box2["xmin"])
    y1 = max(box1["ymin"], box2["ymin"])
    x2 = min(box1["xmax"], box2["xmax"])
    y2 = min(box1["ymax"], box2["ymax"])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1["xmax"] - box1["xmin"]) * (box1["ymax"] - box1["ymin"])
    area2 = (box2["xmax"] - box2["xmin"]) * (box2["ymax"] - box2["ymin"])

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def is_contained(inner: dict, outer: dict) -> bool:
    """
    Check if inner box is fully or mostly contained within outer box.
    Args:
        inner (dict): Inner box with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        outer (dict): Outer box with keys 'xmin', 'ymin', 'xmax', 'ymax'.
    Returns:
        bool: True if inner is contained in outer, False otherwise.
    """
    # Check if inner is completely inside outer
    if (
        inner["xmin"] >= outer["xmin"]
        and inner["xmax"] <= outer["xmax"]
        and inner["ymin"] >= outer["ymin"]
        and inner["ymax"] <= outer["ymax"]
    ):
        return True

    # Calculate intersection
    x1 = max(inner["xmin"], outer["xmin"])
    y1 = max(inner["ymin"], outer["ymin"])
    x2 = min(inner["xmax"], outer["xmax"])
    y2 = min(inner["ymax"], outer["ymax"])

    # No intersection
    if x2 <= x1 or y2 <= y1:
        return False

    inter_width = x2 - x1
    inter_height = y2 - y1
    inter_area = inter_width * inter_height

    inner_width = inner["xmax"] - inner["xmin"]
    inner_height = inner["ymax"] - inner["ymin"]
    inner_area = inner_width * inner_height

    outer_width = outer["xmax"] - outer["xmin"]
    outer_height = outer["ymax"] - outer["ymin"]
    outer_area = outer_width * outer_height

    if inner_area == 0 or outer_area == 0:
        return False

    # Check if most of inner is inside outer (>50% of inner's area)
    containment_ratio = inter_area / inner_area
    if containment_ratio > 0.5:
        return True

    # Check if most of outer is inside inner (>50% of outer's area) - reverse case
    reverse_containment_ratio = inter_area / outer_area
    if reverse_containment_ratio > 0.5:
        return True

    # Check if boxes overlap significantly (intersection > 30% of smaller box)
    smaller_area = min(inner_area, outer_area)
    if inter_area / smaller_area > 0.3:
        return True

    return False


def merge_overlapping_boxes(
    boxes: list[dict], iou_threshold: float = 0.3
) -> list[dict]:
    """
    Merge overlapping or nearby bounding boxes into single detections.

    Args:
        boxes (list[dict]): List of detection dictionaries with 'bbox', 'class', 'confidence'.
        iou_threshold (float): IoU threshold to consider boxes as overlapping.
            Also merges boxes where one contains the other.

    Returns:
        list[dict]: List of merged detections.
    """
    if not boxes:
        return boxes

    # Keep merging until no more merges are possible
    merged = boxes.copy()
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for i, det1 in enumerate(merged):
            if i in used:
                continue

            current = det1
            for j, det2 in enumerate(merged):
                if i >= j or j in used:
                    continue
                if det1["class"] != det2["class"]:
                    continue

                iou = calc_iou(current["bbox"], det2["bbox"])
                contained = is_contained(det2["bbox"], current["bbox"]) or is_contained(
                    current["bbox"], det2["bbox"]
                )

                if iou > iou_threshold or contained:
                    current = merge_two_boxes(current, det2)
                    used.add(j)
                    changed = True

            new_merged.append(current)
            used.add(i)

        merged = new_merged

    return merged


def load_config(config_path: str) -> tuple[float, float, float, int]:
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        config (dict): Dictionary containing the configuration parameters.
    """
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    Fbin = float(config["Fbin"])  # Frequency resolution of the spectrogram [Hz]
    Tbin = float(config["Tbin"])  # Time resolution of the spectrogram [s]
    Foffset = float(
        config["Foffset"]
    )  # Frequency of the bottom pixel in the spectrogram [Hz]
    Npxs = int(config["Npxs"])  # Total number of pixels in spectrogram height

    return Tbin, Fbin, Foffset, Npxs


def test_model(
    model_path: str,
    data_path: str,
    conf: float = 0.50,
    merge_iou: float = 0.25,
    batch_size: int = 16,
    device: str = "cpu",
) -> DetMetrics:
    """
    Test the YOLO model on the specified dataset.

    Args:
        model_path (str): Path to the YOLO model file.
        data_path (str): Path to the dataset for testing.
        conf (float): Confidence threshold for detection.
        merge_iou (float): IoU threshold for NMS.
        batch_size (int): Batch size for testing.
        device (str): Device to run the model on (cpu or cuda).

    Returns:
        results: list of Detection results from the model.
    """

    logger.info(
        f"Testing model: {model_path} on data: {data_path} with conf: {conf}, iou: {merge_iou}, batch_size: {batch_size}, device: {device}"
    )
    # Load the YOLO model
    model = YOLO(model_path)

    # Run the test
    results = model.predict(
        source=data_path,
        conf=conf,
        iou=merge_iou,
        batch=batch_size,
        device=device,
        save=False,
        verbose=False,
        end2end=False,
        nms=True,
    )

    return results


def paint_results(
    results: DetMetrics, save_path: str = "../results", merge_iou: float = 0.3
) -> None:
    """
    Paint the detection results on the images and save them.

    Args:
        results: list of Detection results from the model.
        save_path (str): Directory to save the painted images.
        merge_iou (float): IoU threshold to merge overlapping boxes.
    """

    logger.info(f"Painting results and saving to: {save_path}")

    colors_classes = {
        "w": (0, 255, 0),  # Green for whistles
        "n": (0, 0, 255),  # Red for noise
    }

    # Draw rectangles and labels on the images
    for result in results:
        image_path = result.path
        img = cv2.imread(image_path)

        # Convert boxes to list of dicts for merging
        boxes_list = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            boxes_list.append(
                {
                    "class": result.names[cls],
                    "confidence": conf,
                    "bbox": {
                        "xmin": int(xyxy[0]),
                        "ymin": int(xyxy[1]),
                        "xmax": int(xyxy[2]),
                        "ymax": int(xyxy[3]),
                    },
                }
            )

        # Merge overlapping boxes
        merged_boxes = merge_overlapping_boxes(boxes_list, iou_threshold=merge_iou)

        # Draw merged boxes
        for det in merged_boxes:
            bbox = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class"]

            # Draw rectangle
            img = cv2.rectangle(
                img,
                (bbox["xmin"], bbox["ymin"]),
                (bbox["xmax"], bbox["ymax"]),
                (0, 255, 0),
                2,
            )

            # Put label
            label = f"{cls_name}: {conf:.2f}"
            img = cv2.putText(
                img,
                label,
                (bbox["xmin"], bbox["ymin"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors_classes.get(cls_name, (255, 255, 255)),
                2,
            )

        base_name = os.path.basename(image_path)
        output_path = os.path.join(save_path, base_name)
        cv2.imwrite(output_path, img)


def save_jsons(
    results: DetMetrics, save_path: str = "../results", merge_iou: float = 0.3
) -> None:
    """
    Save the detection results to a json files.
    Args:
        results: list of Detection results from the model.
        save_path (str): Path to save the json results files.
        merge_iou (float): IoU threshold to merge overlapping boxes.
    """

    logger.info(f"Saving results to json files in: {save_path}")

    for result in results:
        json_path = os.path.join(
            save_path, os.path.splitext(os.path.basename(result.path))[0] + ".json"
        )
        json_result = []

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

        # Merge overlapping boxes
        merged_result = merge_overlapping_boxes(json_result, iou_threshold=merge_iou)

        with open(json_path, "w") as f:
            json.dump(merged_result, f, indent=4)


def files_list_creator(
    folder: str,
    filesList_extension: str = "",
    filesList_contains: Optional[str] = None,
) -> list[str]:
    """
    Create a list of files inside a folder (and its subfolders) that match
    a given extension and contain specific substrings in their names.

    Args:
        folder (str): Path to the root folder to search.
        filesList_extension (str): File extension to filter by (e.g. ".txt", ".jpg").
        filesList_contains (str | None): Substring that must be present
            in the filename. If None, no substring filtering is applied.

    Returns:
        list[str]: List of full file paths that match the specified criteria.
    """
    filesList = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            # Check extension
            if filesList_extension and not file.endswith(filesList_extension):
                continue
            # Check substring
            if filesList_contains and filesList_contains not in file:
                continue
            filesList.append(os.path.join(root, file))

    return filesList


def calc_hist(
    data: np.ndarray,
    BinRes: float,
    prob: bool = False,
    Ythreshold: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[int], Optional[float]]:
    """
    Compute a histogram from the given data with a specified bin resolution.

    Args:
        data (np.ndarray): Input data for the histogram calculation.
        BinRes (float): Bin resolution (width of each histogram bin).
        prob (bool): If True, the histogram values are returned as percentages
            instead of raw counts. Default is False.
        Ythreshold (float | None): Cumulative percentage threshold (0-100) used
            to determine the first bin where the cumulative histogram reaches this
            value. If None, no threshold calculation is performed.

    Returns:
        tuple[np.ndarray, np.ndarray, Optional[int], Optional[float]]:
            Yvalue (np.ndarray): Histogram counts or percentages per bin.
            Nbins (np.ndarray): Centers of the histogram bins.
            nTh (int | None): Index of the first bin reaching the cumulative threshold.
            thVal (float | None): Cumulative value at the threshold bin.
    """
    data = np.array(data)
    if len(data) == 0:
        return np.array([]), np.array([]), None, None

    Nbins = np.arange(0, np.max(data) + BinRes, BinRes) + BinRes / 2
    Yvalue = np.zeros(len(Nbins))

    # Fill histogram
    for i in range(len(Nbins)):
        lim_down = Nbins[i] - BinRes / 2
        lim_up = Nbins[i] + BinRes / 2
        Yvalue[i] = np.sum((data > lim_down) & (data <= lim_up))

    # Convert to percentage if necessary
    if prob:
        total = np.sum(Yvalue)
        if total > 0:
            Yvalue = Yvalue / total * 100

    # Threshold value calculation (optional)
    nTh = None
    thVal = None
    if Ythreshold is not None and np.sum(Yvalue) > 0:
        Ycumsum = np.cumsum(Yvalue)
        idx = np.where(Ycumsum >= Ythreshold)[0]
        if len(idx) > 0:
            nTh = idx[0]
            thVal = Ycumsum[nTh]

    return Yvalue, Nbins, nTh, thVal


def save_hist(
    Yvalue: np.ndarray,
    Nbins: np.ndarray,
    FontSize: int,
    title_str: str,
    Xlabel_str: str,
    output_results: str,
    HistName: str,
) -> None:
    """
    Plot and save a histogram with a Gaussian fit overlay.

    Args:
        Yvalue (np.ndarray): Histogram values (counts or percentages).
        Nbins (np.ndarray): Centers of the histogram bins.
        FontSize (int): Font size used for labels, title, and ticks.
        title_str (str): Title of the histogram plot.
        Xlabel_str (str): Label for the x-axis.
        output_results (str): Path to the directory where the figure will be saved.
        HistName (str): Name of the output histogram file (without extension).

    Returns:
        None
    """
    if len(Nbins) < 2:
        return None

    BinRes = Nbins[1] - Nbins[0]
    mu = np.sum(Yvalue * Nbins) / np.sum(Yvalue) if np.sum(Yvalue) > 0 else 0
    variance = (
        np.sum(Yvalue * (Nbins - mu) ** 2) / np.sum(Yvalue) if np.sum(Yvalue) > 0 else 0
    )
    sigma = np.sqrt(variance)
    x_vals = np.linspace(Nbins[0], Nbins[-1], 1000)
    gaussian_vals = scp.stats.norm.pdf(x_vals, mu, sigma) * np.sum(Yvalue) * BinRes

    plt.figure(figsize=(8, 8))
    plt.bar(Nbins, Yvalue, width=BinRes * 0.85, color="skyblue", edgecolor="k")
    plt.plot(
        x_vals,
        gaussian_vals,
        color="red",
        lw=2,
        label=rf"$\mu$: {mu:.2f}, $\sigma$: {sigma:.2f}",
    )
    plt.xlabel(Xlabel_str, fontsize=FontSize)
    plt.ylabel("Samples [%]", fontsize=FontSize)
    plt.xticks(rotation=35)
    plt.legend(loc="upper right", ncol=1, fontsize=FontSize - 2)
    plt.title(title_str, fontsize=FontSize)
    nonzero_idx = np.where(Yvalue != 0)[0]
    if len(nonzero_idx) > 0:
        plt.xlim(
            Nbins[nonzero_idx[0]] - BinRes / 2, Nbins[nonzero_idx[-1]] + BinRes / 2
        )
    plt.tick_params(axis="both", which="major", labelsize=FontSize)
    plt.tick_params(axis="both", which="minor", labelsize=FontSize)
    plt.tight_layout()
    png_file = os.path.join(output_results, f"{HistName}.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    return None


def get_bbox_params(
    item: dict,
    Tbin: float,
    Fbin: float,
    Foffset: float,
    num_pixels: int,
) -> tuple[float, float, float, float, float]:
    """
    Convert bounding box pixel coordinates into time-frequency parameters.

    Args:
        item (dict): Detection item containing the bounding box and confidence
            score. Expected structure:
                item["bbox"] = {"xmin", "xmax", "ymin", "ymax"}
                item["confidence"] = float
        Tbin (float): Time resolution per pixel [s].
        Fbin (float): Frequency resolution per pixel [Hz].
        Foffset (float): Base frequency offset [Hz].
        num_pixels (int): Total number of pixels in the frequency axis.

    Returns:
        tuple[float, float, float, float, float]:
            Conf (float): Detection confidence.
            Tini (float): Initial time of the detection [s].
            Tdur (float): Duration of the detection [s].
            Fmin (float): Minimum frequency of the detection [Hz].
            Fmax (float): Maximum frequency of the detection [Hz].
    """
    xmin = item["bbox"]["xmin"]
    xmax = item["bbox"]["xmax"]
    ymin = item["bbox"]["ymin"]
    ymax = item["bbox"]["ymax"]

    Conf = item["confidence"]
    Tini = xmin * Tbin
    Tdur = (xmax - xmin) * Tbin
    Fmin = Foffset + (num_pixels - ymax) * Fbin
    Fmax = Foffset + (num_pixels - ymin) * Fbin
    return Conf, Tini, Tdur, Fmin, Fmax


def plot_WRSresults(
    results_df: pd.DataFrame,
    font_size: int,
    output_results: str,
    file_name_output: str,
) -> None:
    """
    Generate and save boxplots summarizing whistle time and frequency statistics.

    Args:
        results_df (pd.DataFrame): DataFrame containing whistle detection
            parameters. It must include the columns 'Tdur', 'Fmin', 'Fmax', and 'Fdur'.
        font_size (int): Font size used for labels, titles, and ticks.
        output_results (str): Path to the directory where the figure will be saved.
        file_name_output (str): Name of the output figure file (without extension).

    Returns:
        None
    """
    if results_df.empty:
        logger.warning("No data to plot.")
        return None

    sns.set_style("whitegrid")
    sns.set_palette("dark")

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 2]}
    )
    fig.suptitle(f"{len(results_df)} whistles", fontsize=font_size, fontweight="bold")
    plt.subplots_adjust(top=0.88, wspace=0.3)

    # Left axes: Tdur box plot
    sns.boxplot(
        x=["Tdur"] * len(results_df),
        y=results_df["Tdur"],
        ax=axes[0],
        width=0.3,
        color="white",
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(
            marker="o", markerfacecolor="black", markersize=4, linestyle="none"
        ),
    )
    axes[0].set_xlabel("", fontsize=font_size)
    axes[0].set_ylabel("Time [s]", fontsize=font_size)
    axes[0].tick_params(axis="both", which="major", labelsize=font_size)

    # Right axes: Fmin, Fmax, Fdur box plot
    freq_data = results_df[["Fmin", "Fmax", "Fdur"]] * 1e-3
    df_long = freq_data.melt(var_name="FrequencyType", value_name="Frequency_kHz")
    sns.boxplot(
        x="FrequencyType",
        y="Frequency_kHz",
        data=df_long,
        ax=axes[1],
        width=0.3,
        color="white",
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(
            marker="o", markerfacecolor="black", markersize=4, linestyle="none"
        ),
    )
    axes[1].set_xlabel("", fontsize=font_size)
    axes[1].set_ylabel("Frequency [kHz]", fontsize=font_size)
    axes[1].tick_params(axis="both", which="major", labelsize=font_size)

    plt.tight_layout()
    png_file = os.path.join(output_results, f"{file_name_output}.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    return None
