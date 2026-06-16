# CLAUDE.md - Whistles Recognition System (WRS) Development Guide

> Documentation for understanding, developing, and testing the Whistles Recognition System.

---

## 📋 Quick Start

### Running the Application

**Detect whistles in images:**
```bash
python run_wrs.py application \
  --data_folder ./images \
  --output_results ./detections \
  --conf 0.5 \
  --merge_iou 0.3
```

**Analyze detection results:**
```bash
python run_wrs.py results \
  --data_folder ./detections \
  --output_results ./analysis \
  --output_hist
```

### Running Tests

**All tests:**
```bash
pytest -v
```

**Specific test categories:**
```bash
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v              # Integration tests
pytest tests/validation/ -v               # Results validation
```

**With coverage report:**
```bash
pytest --cov=src --cov=scripts --cov-report=term-missing
```

---

## 🏗️ Project Structure

```
Whistles-Recognition-System/
├── run_wrs.py                    # CLI launcher (entry point)
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
├── pyproject.toml                 # Project metadata
│
├── src/
│   ├── __init__.py
│   ├── config.yaml                # Spectrogram parameters (Fbin, Tbin, etc.)
│   └── utils.py                   # Core utility functions
│       ├── merge_two_boxes()
│       ├── calc_iou()
│       ├── is_contained()
│       ├── merge_overlapping_boxes()
│       ├── files_list_creator()
│       ├── load_config()
│       ├── test_model()
│       ├── paint_results()
│       ├── save_jsons()
│       ├── get_bbox_params()
│       └── ...
│
├── scripts/
│   ├── __init__.py
│   ├── WRSapplication.py          # Whistle detection (YOLO inference)
│   └── WRSresults.py              # Results analysis & statistics
│
├── tests/
│   ├── conftest.py                # Shared pytest fixtures
│   ├── unit/
│   │   ├── test_utils.py          # Unit tests for src/utils.py
│   │   ├── test_wrsapplication.py # Tests for WRSapplication
│   │   └── test_wrsresults.py     # Tests for WRSresults
│   ├── integration/
│   │   └── test_pipeline.py       # End-to-end pipeline tests
│   ├── validation/
│   │   └── test_results.py        # Output validation tests
│   └── results_validation/        # Sample outputs for verification
│
├── models/
│   └── best_exp20.pt              # YOLO model (YOLOv8 format)
│
├── images/                        # Sample images for testing
│
├── results/                       # Output storage
│
└── .github/
    └── workflows/
        └── ci.yml                 # GitHub Actions CI configuration
```

---

## 🔄 Workflow Overview

### 1. **Whistle Detection Phase** (`WRSapplication.py`)

```
Input Images
    ↓
[YOLO Model Inference]
    ├─ Load model from ./models/best_exp20.pt
    ├─ Batch predict on images
    ├─ Merge overlapping detections (IOU-based)
    ↓
JSON Outputs + PNG Visualizations
```

**Key Parameters:**
- `--conf`: Confidence threshold (default: 0.5)
- `--merge_iou`: IoU threshold for merging (default: 0.3)
- `--device`: GPU/CPU selection (default: cpu)

**Output:**
- `{image_name}.json`: Detection boxes [class, confidence, bbox]
- `{image_name}.png`: Annotated images

### 2. **Analysis Phase** (`WRSresults.py`)

```
JSON Detections
    ↓
[Extract Whistle Parameters]
    ├─ Filter class == "w" (whistles only)
    ├─ Convert pixel coords → time/frequency
    ├─ Calculate Tdur, Fmin, Fmax, Fdur
    ↓
CSV Report + Histograms + Boxplots
```

**Config Parameters** (`src/config.yaml`):
- `Fbin`: Frequency resolution [Hz/pixel]
- `Tbin`: Time resolution [s/pixel]
- `Foffset`: Base frequency of spectrogram [Hz]
- `Npxs`: Total frequency pixels

**Output:**
- `{name}.csv`: Whistle statistics table
- `{name}_*.png`: Histograms for each parameter

---

## 🧪 Testing Strategy

### Unit Tests (`tests/unit/`)

**`test_utils.py`** - Core algorithm testing
- IoU calculation (identical, overlapping, non-overlapping)
- Box merging (containment, union computation)
- File discovery and filtering
- Config loading
- Histogram generation

**`test_wrsapplication.py`** - Detection workflow
- File listing integration
- Output directory handling
- JSON result structure validation
- IoU threshold impact

**`test_wrsresults.py`** - Analysis workflow
- JSON parsing
- Parameter extraction
- DataFrame construction
- CSV formatting

### Integration Tests (`tests/integration/`)

**`test_pipeline.py`** - Complete workflows
- JSON → CSV pipeline
- Box merging + analysis chain
- Histogram generation
- Multi-file aggregation
- Data format conversions

### Validation Tests (`tests/validation/`)

**`test_results.py`** - Output correctness
- Detection format validation
- Merged box consistency
- Physical validity of parameters
- Data integrity through pipeline
- Metadata preservation

---

## 📊 Testing Coverage

Expected test coverage by component:

```
src/utils.py
  └─ merge_two_boxes()          [100%]
  └─ calc_iou()                 [100%]
  └─ is_contained()             [100%]
  └─ merge_overlapping_boxes()  [100%]
  └─ files_list_creator()       [100%]
  └─ calc_hist()                [100%]
  └─ load_config()              [100%]
  └─ get_bbox_params()          [100%]

scripts/WRSapplication.py
  └─ run()                      [integration tests]

scripts/WRSresults.py
  └─ run()                      [integration tests]
```

---

## 🛠️ Development Workflow

### Adding a New Feature

1. **Write tests first** (TDD):
   ```bash
   # Create test_new_feature.py in appropriate directory
   ```

2. **Implement the feature** in `src/utils.py` or script

3. **Run tests**:
   ```bash
   pytest tests/unit/test_new_feature.py -v
   ```

4. **Check coverage**:
   ```bash
   pytest --cov=src tests/unit/ --cov-report=term-missing
   ```

### Debugging a Test

```bash
# Run with detailed output
pytest tests/unit/test_utils.py::TestIOUCalculation -vv

# Drop into debugger on failure
pytest tests/unit/test_utils.py -vv --pdb

# Show print statements
pytest tests/unit/test_utils.py -s
```

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Test application workflow
python run_wrs.py application --data_folder ./images --output_results ./test_out

# Test analysis
python run_wrs.py results --data_folder ./test_out --output_results ./analysis

# Run all tests
pytest -v --tb=short
```

---

## 🔐 Key Algorithms

### IoU (Intersection over Union)

Used to determine if two bounding boxes overlap:

```
IoU = Intersection Area / Union Area
      = Overlap / (Box1 + Box2 - Overlap)

Range: [0, 1]
  0 = no overlap
  1 = identical boxes
```

### Box Merging

Merges overlapping boxes if:
1. **IoU > threshold** (spatial overlap), OR
2. **One box contains >50% other box** (containment)

Merged box properties:
- Bounds: Union of both boxes
- Confidence: Max of both confidences
- Class: Inherited from first box (requires same class)

### Whistle Parameter Extraction

Converts pixel coordinates to physical parameters:

```
Time (s)      = xmin × Tbin
Duration (s)  = (xmax - xmin) × Tbin
Freq Min (Hz) = Foffset + (Npxs - ymax) × Fbin
Freq Max (Hz) = Foffset + (Npxs - ymin) × Fbin
```

---

## 🐛 Troubleshooting

### Tests Failing

```bash
# Check Python version (3.9+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Clear cache
rm -rf __pycache__ .pytest_cache

# Run tests again
pytest -v
```

### Model Not Found

```bash
# Check model path
ls -la models/best_exp20.pt

# Update argument in CLI
python run_wrs.py application --model ./models/best_exp20.pt --data_folder ./images
```

### ImportError: No module named 'src'

```bash
# Ensure you're running from project root
pwd
# Should be: .../Whistles-Recognition-System

# Or run with module flag
python -m pytest tests/
```

---

## 📈 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

```yaml
On: push, pull_request
├─ Test (Python 3.11, 3.12)
│  ├─ Install dependencies
│  ├─ Run pytest (all tests)
│  └─ Generate coverage report
├─ Lint (pre-commit hooks)
│  ├─ YAML validation
│  ├─ File formatting
│  └─ Code consistency
└─ [Optional] Publish coverage to Codecov
```

**To trigger locally:**
```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

---

## 📚 References

- **YOLO Detection**: https://docs.ultralytics.com/modes/predict/
- **IoU Metric**: https://en.wikipedia.org/wiki/Jaccard_index
- **matplotlib/seaborn**: For visualization
- **pandas**: For data manipulation

---

## ✅ Next Steps for Contributors

1. ✅ Read this CLAUDE.md
2. ✅ Run `pytest` to verify setup
3. ✅ Pick an issue from TODO list
4. ✅ Create tests for new feature
5. ✅ Implement feature
6. ✅ Ensure all tests pass
7. ✅ Create pull request

**Happy coding! 🎵🔍**
