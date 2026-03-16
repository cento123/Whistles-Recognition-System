# %% Imports
import json
import os
import logging
import os
import json
import logging
import cv2
import numpy as np
import scipy as scp 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% test_model function():
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

    logger.info(
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
        verbose=False,
    )

    return results

# %% paint_results function():
def paint_results(results: DetMetrics, save_path: str = "../results") -> None:
    """
    Paint the detection results on the images and save them.
    Args:
        results: list of Detection results from the model.
        save_path (str): Directory to save the painted images.
    """

    logger.info(f"Painting results and saving to: {save_path}")

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Draw rectangles and labels on the images
    for result in results:
        image_path = result.path
        img = cv2.imread(image_path)

        for box in result.boxes:
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

# %% save_jsons function():
def save_jsons(results: DetMetrics, save_path: str = "../results") -> None:
    """
    Save the detection results to a json files.
    Args:
        results: list of Detection results from the model.
        save_path (str): Path to save the json results files.
    """

    logger.info(f"Saving results to json files in: {save_path}")

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

# %% filesListCreator function():
def files_list_creator(folder: str, filesList_extension: str = "", filesList_contains: Optional[List[str]] = None, ) -> List[str]: 
    """ 
    Create a list of files inside a folder (and its subfolders) that match 
    a given extension and contain specific substrings in their names. 
    
    Args: 
        folder (str): Path to the root folder to search. 
        filesList_extension (str): File extension to filter by (e.g. ".txt", ".jpg"). 
        filesList_contains (List[str] | None): List of substrings that must be present 
            in the filename. If None, no substring filtering is applied. 
            
    Returns: 
        List[str]: List of full file paths that match the specified criteria. 
    """
    # Initialize an empty list to store the matched files
    filesList = []
    # Set the default value for filesList_contains to an empty list if it's None
    if filesList_contains is None:
        filesList_contains = []
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Check if the file ends with the specified extension and contains all specified substrings
            if file.endswith(filesList_extension) and all(sub in file for sub in filesList_contains):
                filesList.append(os.path.join(root, file))
    return filesList

# %% calc_hist function:
def calc_hist(data: np.ndarray, BinRes: float, prob: bool = False, Ythreshold: Optional[float] = None, ) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[float]]: 
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
        Tuple[np.ndarray, np.ndarray, Optional[int], Optional[float]]: 
            Yvalue (np.ndarray): Histogram counts or percentages per bin. 
            Nbins (np.ndarray): Centers of the histogram bins. 
            nTh (int | None): Index of the first bin reaching the cumulative 
                threshold. None if no threshold is provided or reached. 
            thVal (float | None): Cumulative value at the threshold bin. 
                None if no threshold is provided or reached. 
    """
    # Function implementation goes here  
    data = np.array(data)
    Nbins = np.arange(0, np.max(data) + BinRes, BinRes) + BinRes/2
    Yvalue = np.zeros(len(Nbins))
    # Fill histogram
    for i in range(len(Nbins)):
        lim_down = Nbins[i] - BinRes/2
        lim_up = Nbins[i] + BinRes/2
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

# %% save_hist function:
def save_hist(Yvalue: np.ndarray, Nbins: np.ndarray, FontSize: int, title_str: str, Xlabel_str: str, output_results: str, HistName: str, ) -> None: 
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
    # Compute mean and std for Gaussian overlay
    BinRes = Nbins[1]-Nbins[0]
    mu = np.sum(Yvalue * Nbins) / np.sum(Yvalue) if np.sum(Yvalue) > 0 else 0
    variance = np.sum(Yvalue * (Nbins - mu)**2) / np.sum(Yvalue) if np.sum(Yvalue) > 0 else 0
    sigma = np.sqrt(variance)
    x_vals = np.linspace(Nbins[0], Nbins[-1], 1000)
    gaussian_vals = scp.stats.norm.pdf(x_vals, mu, sigma) * np.sum(Yvalue) * BinRes

    plt.figure(figsize=(8,8))
    plt.bar(Nbins, Yvalue, width=BinRes*0.85, color='skyblue', edgecolor='k')
    plt.plot(x_vals, gaussian_vals, color='red', lw=2,label=rf'$\mu$: {mu:.2f}, $\sigma$: {sigma:.2f}')
    plt.xlabel(Xlabel_str, fontsize=FontSize)
    plt.ylabel('Samples [%]', fontsize=FontSize)
    plt.xticks(rotation=35)
    plt.legend(loc='upper right', ncol=1, fontsize=FontSize-2)
    plt.title(title_str, fontsize=FontSize)
    nonzero_idx = np.where(Yvalue != 0)[0]
    if len(nonzero_idx) > 0:
        plt.xlim(Nbins[nonzero_idx[0]] - BinRes/2,
                Nbins[nonzero_idx[-1]] + BinRes/2)
    plt.tick_params(axis='both', which='major', labelsize=FontSize)
    plt.tick_params(axis='both', which='minor', labelsize=FontSize)
    plt.tight_layout()
    png_file = os.path.join(output_results, f"{HistName}.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    return None

# %% get_bbox_params function: 
def get_bbox_params(item: dict, Tbin: float, Fbin: float, Foffset: float, num_pixels: int,) -> tuple[float, float, float, float, float]:
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
        Tuple[float, float, float, float, float]: 
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
    Tini = xmin*Tbin
    Tdur = (xmax-xmin)*Tbin
    Fmin = Foffset + (num_pixels - ymax) * Fbin
    Fmax = Foffset + (num_pixels - ymin) * Fbin
    return Conf, Tini, Tdur, Fmin, Fmax

# %% plot_WRSresults function: 
def plot_WRSresults(results_df: pd.DataFrame, font_size: int, output_results: str, file_name_output: str, ) -> None: 
    """ 
    Generate and save boxplots summarizing whistle time and frequency statistics. 
    The function creates a figure with two subplots: 
        1. A boxplot showing the distribution of whistle durations (Tdur). 
        2. Boxplots showing the distributions of minimum frequency (Fmin), maximum frequency (Fmax), and frequency bandwidth (Fdur). 
        
    Args: 
        results_df (pd.DataFrame): DataFrame containing whistle detection 
            parameters. It must include the columns 'Tdur', 'Fmin', 'Fmax', and 'Fdur'. 
        font_size (int): Font size used for labels, titles, and ticks. 
        output_results (str): Path to the directory where the figure will be saved. 
        file_name_output (str): Name of the output figure file (without extension). 
    
    Returns: 
        None 
    """
    sns.set_style("whitegrid")
    sns.set_palette("dark")
    # Adjust relative widths: left subplot 1/3, right subplot 2/3
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle(f"{len(results_df)} whistles", fontsize=font_size, fontweight='bold')
    plt.subplots_adjust(top=0.88, wspace=0.3)  # add space between subplots
    # Left axes: Tdur box plot (single box)
    sns.boxplot(
        x=['Tdur']*len(results_df),
        y=results_df['Tdur'],
        ax=axes[0],
        width=0.3,
        color='white',
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=4, linestyle='none')
    )
    axes[0].set_xlabel('', fontsize=font_size)
    axes[0].set_ylabel('Time [s]', fontsize=font_size)
    axes[0].tick_params(axis='both', which='major', labelsize=font_size)
    axes[0].tick_params(axis='both', which='minor', labelsize=font_size)
    # Right axes: Fmin, Fmax, Fdur box plot (3 boxes)
    freq_data = results_df[['Fmin', 'Fmax', 'Fdur']] * 1e-3
    df_long = freq_data.melt(var_name='FrequencyType', value_name='Frequency_kHz')
    sns.boxplot(
        x='FrequencyType',
        y='Frequency_kHz',
        data=df_long,
        ax=axes[1],
        width=0.3,
        color='white',
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=4, linestyle='none')
    )
    axes[1].set_xlabel('', fontsize=font_size)
    axes[1].set_ylabel('Frequency [kHz]', fontsize=font_size)
    axes[1].tick_params(axis='both', which='major', labelsize=font_size)
    axes[1].tick_params(axis='both', which='minor', labelsize=font_size)
    plt.tight_layout()
    # Save figure
    png_file = os.path.join(output_results, f"{file_name_output}.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    return None
