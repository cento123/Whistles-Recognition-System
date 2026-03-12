# %% Imports
import json
import os
import logging
import os
import json
import logging
import numpy as np
import scipy as scp 
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

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
def filesListCreator(folder, filesList_extension='', filesList_contains=None):
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

# %% calcHist function:
def calcHist(data, BinRes, prob=False, Ythreshold=None):
    '''
    Create a histogram from given data.

    Summary:
    This function creates a histogram from the given data with specified bin resolution.
    Optionally, it can plot the histogram and compute the percentage of samples within a threshold.

    Parameters:
    - data (numpy.ndarray): Data for the analysis.
    - BinRes (float): Bin resolution for the histogram.
    - prob (bool, optional): If True, the result is obtained in percentage (not in counts). Default is False.
    - Ythreshold (float, optional): Cumulative percentage threshold for calculating nTh and thVal (0-100). Default: None.

    Returns:
    - Yvalue (numpy.ndarray): Histogram counts or percentages.
    - Nbins (numpy.ndarray): Bin edges.
    - nTh (float): Index of the threshold bin (if Ythreshold is provided).
    - thVal (float): Percentage of samples up to the threshold (if Ythreshold is provided).

    Application:
    Yvalue, Nbins, nTh, thVal = calcHist(data, BinRes, prob, Ythreshold)

    Dependencies:
    - numpy: For array manipulation.

    Created/Last modified: 2026-02-23
    '''
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

# %% saveHist function:
def saveHist(Yvalue, Nbins, FontSize, Nwhistles, Xlabel_str, output_results, FileName_output):
    # Compute mean and std for Gaussian overlay
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
    plt.title(f"{Nwhistles} whistles\n{col}", fontsize=FontSize)
    nonzero_idx = np.where(Yvalue != 0)[0]
    if len(nonzero_idx) > 0:
        plt.xlim(Nbins[nonzero_idx[0]] - BinRes/2,
                Nbins[nonzero_idx[-1]] + BinRes/2)
    plt.tick_params(axis='both', which='major', labelsize=FontSize)
    plt.tick_params(axis='both', which='minor', labelsize=FontSize)
    plt.tight_layout()
    png_file = os.path.join(output_results, f"{FileName_output}_{col}.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    return None

# %% GetItemParams function: 
def GetItemParams(item,Tpx,Fpx,Fpx_0,Npxs):
    xmin = item["bbox"]["xmin"]
    xmax = item["bbox"]["xmax"]
    ymin = item["bbox"]["ymin"]
    ymax = item["bbox"]["ymax"]

    Conf = item["confidence"]
    Tini = xmin*Tpx
    Tdur = (xmax-xmin)*Tpx
    Fmin = Fpx_0 + (Npxs - ymax) * Fpx
    Fmax = Fpx_0 + (Npxs - ymin) * Fpx
    return Conf, Tini, Tdur, Fmin, Fmax

# %% saveWRSreults function: 
def saveWRSreults(Results_df, FontSize, output_results, FileName_output):
    sns.set_style("whitegrid")
    sns.set_palette("dark")
    # Adjust relative widths: left subplot 1/3, right subplot 2/3
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle(f"{len(Results_df)} whistles", fontsize=FontSize, fontweight='bold')
    plt.subplots_adjust(top=0.88, wspace=0.3)  # add space between subplots
    # Left axes: Tdur box plot (single box)
    sns.boxplot(
        x=['Tdur']*len(Results_df),
        y=Results_df['Tdur'],
        ax=axes[0],
        width=0.3,
        color='white',
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=4, linestyle='none')
    )
    axes[0].set_xlabel('', fontsize=FontSize)
    axes[0].set_ylabel('Time [s]', fontsize=FontSize)
    axes[0].tick_params(axis='both', which='major', labelsize=FontSize)
    axes[0].tick_params(axis='both', which='minor', labelsize=FontSize)
    # Right axes: Fmin, Fmax, Fdur box plot (3 boxes)
    freq_data = Results_df[['Fmin', 'Fmax', 'Fdur']] * 1e-3
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
    axes[1].set_xlabel('', fontsize=FontSize)
    axes[1].set_ylabel('Frequency [kHz]', fontsize=FontSize)
    axes[1].tick_params(axis='both', which='major', labelsize=FontSize)
    axes[1].tick_params(axis='both', which='minor', labelsize=FontSize)
    plt.tight_layout()
    # Save figure
    png_file = os.path.join(output_results, f"{FileName_output}.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    return None