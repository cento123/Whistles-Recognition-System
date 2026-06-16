# End-to-End Integration Tests Guide

This guide explains how to run the end-to-end (E2E) integration tests that verify the complete WRS pipeline.

---

## 📋 Test Files

### 1. **test_e2e_local.py** (Recommended)
Uses **local files** from your project directory. Fastest to run.

### 2. **test_e2e_pipeline.py**
Enhanced version with optional Google Drive integration and detailed logging.

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install test dependencies
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Run E2E Tests with Local Files

```bash
# Run all E2E tests
pytest tests/integration/test_e2e_local.py -v -s

# Run specific test
pytest tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_complete_workflow -v -s

# Run with markers
pytest -m "integration and slow" -v
```

### Run Only with Local Files (Fastest)
```bash
pytest tests/integration/test_e2e_local.py::TestE2ELocalPipeline -v --tb=short
```

---

## 📥 Using Google Drive Data (Optional)

### Download Test Data

If you don't have local test data, download from:
[Google Drive Test Data](https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU?usp=sharing)

**What to download:**
- `best_exp20.pt` → Save to `./models/`
- Sample images → Save to `./images/`

### Manual Download Steps

```bash
# Option 1: Using gdown CLI
pip install gdown

# Download folder (first copy the folder ID)
gdown --folder https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU -O ./data

# Option 2: Manual Download
# 1. Visit the link above
# 2. Download files to local directories:
#    - models/best_exp20.pt
#    - images/*.png
```

---

## ⚙️ Test Structure

### TestE2ELocalPipeline

#### 1. `test_e2e_wrsapplication_flow()`
Tests the **detection phase**:
- ✅ Loads YOLO model
- ✅ Processes image with detection
- ✅ Generates JSON detection files
- ✅ Generates annotated PNG images
- ✅ Validates output structure

#### 2. `test_e2e_wrsresults_flow()`
Tests the **analysis phase**:
- ✅ Runs WRSapplication first
- ✅ Extracts whistle parameters (Tdur, Fmin, Fmax, Fdur)
- ✅ Generates CSV report
- ✅ Validates data integrity
- ✅ Generates optional histograms

#### 3. `test_e2e_complete_workflow()`
**Full pipeline** test:
1. Input image
   ↓
2. Detection phase → JSON + PNG
   ↓
3. Analysis phase → CSV + Histograms
4. Validates all outputs

---

## 🧪 What Each Test Validates

### Detection Phase (WRSapplication)
```
✓ Model loads successfully
✓ Image processing completes
✓ JSON has correct structure:
  - "class": w|n
  - "confidence": 0-1
  - "bbox": {xmin, ymin, xmax, ymax}
✓ PNG annotated image generated
✓ Confidence values in range [0, 1]
✓ Bounding boxes have valid dimensions
```

### Analysis Phase (WRSresults)
```
✓ JSON parsing correct
✓ Whistle filtering (class="w")
✓ Parameter extraction:
  - Tdur: Time duration (s)
  - Fmin: Min frequency (Hz)
  - Fmax: Max frequency (Hz)
  - Fdur: Frequency duration (Hz)
✓ CSV file generated with metadata
✓ Histogram images created (optional)
✓ Data types correct
✓ Value ranges valid
```

---

## 📊 Expected Output

After running the complete workflow, you should see:

```
📸 Running WRSapplication (detection phase)...
✅ Detection complete: 1 JSON, 1 PNG
🎯 Total detections: 15

📊 Running WRSresults (analysis phase)...
✅ Analysis complete: 10 whistles analyzed
📈 Generated 4 histogram files

📋 Whistle Statistics:
  - Confidence: 0.50 - 0.99
  - Duration: 0.050s - 0.400s
  - Freq Min: 800 - 8000 Hz
  - Freq Max: 1200 - 8500 Hz

✅ End-to-end workflow completed successfully!
```

---

## 🔧 Troubleshooting

### Test Skipped: "Model not found"
```bash
# Solution: Download model from Drive or check path
ls -la models/best_exp20.pt
```

### Test Skipped: "No test images"
```bash
# Solution: Download test images from Drive or add to ./images/
ls -la images/
```

### JSON parsing fails
```bash
# Check detection JSON format:
cat results_WRSapplication/*.json | python -m json.tool
```

### CSV empty
```bash
# Check if whistles were detected:
python run_wrs.py application --data_folder ./images --conf 0.3  # Lower threshold
```

---

## 🎯 Running Specific Scenarios

### Only Test Detection
```bash
pytest tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_wrsapplication_flow -v
```

### Only Test Analysis
```bash
pytest tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_wrsresults_flow -v
```

### With Detailed Output
```bash
pytest tests/integration/test_e2e_local.py -v -s --tb=long
```

### With Coverage Report
```bash
pytest tests/integration/test_e2e_local.py --cov=src --cov=scripts --cov-report=term-missing
```

---

## 📈 Performance

Typical execution times:

| Test | Duration | Notes |
|------|----------|-------|
| Detection only | 30-60s | YOLO inference |
| Analysis only | 5-10s | CSV generation |
| Complete workflow | 40-70s | Full pipeline |
| Validation checks | <1s | Format only |

---

## ✅ CI/CD Integration

These tests can be run in GitHub Actions or other CI systems:

```bash
# Fast path (validation only)
pytest tests/integration/test_e2e_local.py::TestDetectionValidation -v

# Full path (requires model + images)
pytest tests/integration/test_e2e_local.py -v --timeout=120
```

---

## 📚 Additional Resources

- **CLAUDE.md** - Full development guide
- **TESTING.md** - Quick testing reference
- **tests/README.md** - Test organization details

---

**Need help?** Check the output logs for specific error messages.
