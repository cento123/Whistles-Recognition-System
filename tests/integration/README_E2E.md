# 🧪 End-to-End Integration Tests for WRS

Complete automated tests for the Whistles Recognition System pipeline.

---

## 🚀 Quick Start

```bash
# 1. Download test data from Google Drive (if needed)
python download_test_data.py

# 2. Run all E2E tests
python run_e2e_tests.py

# 3. View detailed results
python run_e2e_tests.py --verbose --test complete
```

---

## 📦 What's Testing

### Test Files

| File | Purpose |
|------|---------|
| `tests/integration/test_e2e_local.py` | ⭐ **Main tests** - Uses local files |
| `tests/integration/test_e2e_pipeline.py` | Advanced tests - Optional Drive integration |
| `E2E_TESTING_COMPLETE.md` | Full documentation |
| `E2E_TESTING_GUIDE.md` | Technical details |

### Test Classes

#### 1. **TestE2ELocalPipeline** (3 tests)

- `test_e2e_wrsapplication_flow()` - Detection
- `test_e2e_wrsresults_flow()` - Analysis
- `test_e2e_complete_workflow()` - Complete pipeline ⭐

#### 2. **TestDetectionValidation** (2 tests)

- `test_detection_json_format()` - JSON validation
- `test_csv_output_format()` - CSV validation

---

## 📥 Download Test Data

### Automatic (Recommended)

```bash
# Install gdown (optional but recommended)
pip install gdown

# Download and setup
python download_test_data.py

# The script will:
# 1. Download from Google Drive
# 2. Copy best_exp20.pt to ./models/
# 3. Copy images to ./images/
# 4. Cleanup temp files
```

### Manual

1. Visit: https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU
2. Download: `best_exp20.pt` and sample images
3. Create directories:
   ```bash
   mkdir -p models images
   ```
4. Place files:
   ```
   ./models/best_exp20.pt
   ./images/*.png
   ```

---

## 🎯 Running Tests

### All Default Tests
```bash
pytest tests/integration/test_e2e_local.py -v -s
```

### Specific Tests

```bash
# Complete workflow (recommended)
pytest tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_complete_workflow -v -s

# Detection only
pytest tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_wrsapplication_flow -v -s

# Analysis only
pytest tests/integration/test_e2e_local.py::TestE2ELocalPipeline::test_e2e_wrsresults_flow -v -s

# Format validation (no heavy files needed)
pytest tests/integration/test_e2e_local.py::TestDetectionValidation -v
```

### With Coverage

```bash
pytest tests/integration/test_e2e_local.py --cov=src --cov=scripts --cov-report=html
open htmlcov/index.html
```

### Quick Script

```bash
python run_e2e_tests.py                      # Default (complete)
python run_e2e_tests.py --test detection     # Detection only
python run_e2e_tests.py --test validation    # Validation only
python run_e2e_tests.py --verbose            # Detailed output
```

---

## 💾 Expected Files

After running tests, you should have:

```
./models/
├── best_exp20.pt          # YOLO model

./images/
├── sample1.png
├── sample2.png
└── ...

# Optional - temp test outputs (auto-cleaned)
# Tests create temp dirs for JSON, PNG, CSV outputs
```

---

## ✅ What Gets Validated

### Phase 1: Detection (WRSapplication)

```
✓ Model loads
✓ Image processes
✓ JSON generated with:
  - class (w/n)
  - confidence (0-1)
  - bbox (xmin, ymin, xmax, ymax)
✓ PNG annotated
✓ Valid dimensions
```

### Phase 2: Analysis (WRSresults)

```
✓ JSON parsed
✓ Whistles filtered (class="w")
✓ Parameters extracted:
  - Tdur: Duration (s)
  - Fmin: Min frequency (Hz)
  - Fmax: Max frequency (Hz)
  - Fdur: Frequency range (Hz)
✓ CSV generated
✓ Metadata included
✓ Histograms created (optional)
✓ Data integrity verified
✓ Value ranges valid
```

---

## 📊 Expected Output

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

| Problem | Solution |
|---------|----------|
| "Model not found" | Run: `python download_test_data.py` |
| "No test images" | Run: `python download_test_data.py` |
| Test skipped | Check models & images folders exist |
| CSV empty | Whistles not detected - try lower `--conf` |
| Import errors | Run: `pip install -r requirements-test.txt` |
| SSL certificate verify failed | Upgrade `certifi`, set `REQUESTS_CA_BUNDLE`, or use manual download mode |

If you get `CERTIFICATE_VERIFY_FAILED`, run in PowerShell:

```powershell
python -m pip install --upgrade certifi
$env:REQUESTS_CA_BUNDLE=(python -c "import certifi;print(certifi.where())")
$env:SSL_CERT_FILE=(python -c "import certifi;print(certifi.where())")
python download_test_data.py
```

---

## ⚙️ Advanced Options

### Command-line Parameters (edit in test)

```python
sys.argv = [
    "--conf", "0.5",        # Detection confidence
    "--merge_iou", "0.3",   # IoU threshold
    "--device", "cpu",      # cpu or cuda
]
```

### Run with GPU (if available)

```python
# In test file, change to:
"--device", "cuda",
```

### Profiles

**Fast** (validation only):
```bash
pytest tests/integration/test_e2e_local.py::TestDetectionValidation -v
```

**Standard** (local files):
```bash
pytest tests/integration/test_e2e_local.py -v -s
```

**Full** (with coverage):
```bash
pytest tests/integration/test_e2e_local.py --cov=src --cov=scripts -v
```

---

## 📚 Documentation

- **E2E_TESTING_COMPLETE.md** - Full guide with all details
- **E2E_TESTING_GUIDE.md** - Technical specifications
- **CLAUDE.md** - Development guide
- **tests/README.md** - Test suite organization

---

## 🎉 Success!

All tests pass when:
- ✅ JSON detections have correct structure
- ✅ PNG images are annotated
- ✅ CSV has all required columns
- ✅ Parameters are physically valid
- ✅ Data integrity is preserved through pipeline

---

## 📧 Support

Check test output with:
```bash
pytest tests/integration/test_e2e_local.py -vv --tb=long
```

For detailed logs, add `-s` flag to see print statements.

---

**Ready to test?** 🚀

```bash
python run_e2e_tests.py
```
