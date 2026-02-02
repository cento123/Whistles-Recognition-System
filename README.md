
# Whistles Recognition System (WRS) Model Testing

This project provides scripts to test a WRS model described in the paper:
'Exploring You Only Look Once v8 for automatic detection of dolphin whistles in spectrograms'

Avaria-Avaria, V., Diego-Tortosa, D., Gallardo, C., Morell-Monzó, S., & Quiroz-Rangel, C. A. (2026). *Exploring You Only Look Once v8 for automatic detection of dolphin whistles in spectrograms*. Bioacoustics, 1–27. https://doi.org/10.1080/09524622.2025.2612276

It is designed for whistle recognition tasks and includes functionality to evaluate model performance and visualize
detection results.

## Features

- Test WRS models on custom datasets
- Visualize and save detection results with bounding boxes and class labels
- Save json results for further analysis
- Configurable parameters via command-line arguments

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the test script from the command line:

```bash
python src/test.py --model <path_to_model.pt> --data <path_to_test_images> --output_results <results_folder>
```

### Optional Arguments

- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--batch_size`: Batch size for testing (default: 16)
- `--device`: Device to run the model (`cpu` or `cuda`, default: `cpu`)

Example:

```bash
python src/test.py --model ../models/best_exp20.pt --data ../images/test --output_results ../results
```

## Output

- Detection results are saved in the specified results folder with bounding boxes and class labels drawn on the images.

## Project Structure

- `src/test.py`: Main script for testing and visualizing YOLO model results
- `models/`: Directory to store trained WRS model weights from "Phase 3: training with all data"
- `images/`: Directory containing test images from main dataset (H1) used in "Phase 1: experimenting with whistles and pings"
- `images/` --> `gt`: Ground truth annotations for test images
- `results/`: Directory to save output results
- `requirements.txt`: Python dependencies

## Images/labels and models
To download images, labels, and models used in the paper for testing this repository, please visit the following link:

https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU?usp=sharing

## License

This project is for research and educational purposes.
