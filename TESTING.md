# WRS: Quick Start & Testing Guide 

## 🚀 Getting Started (5 minutes) 

### 1. Install Dependencies
```bash
pip install -r requirements-test.txt
```

### 2. Run Basic Tests
```bash
pytest tests/ -v --tb=short
```

### 3. Run the Application
```bash
# Detect whistles in images
python run_wrs.py application --data_folder ./images --output_results ./detections

# Analyze results
python run_wrs.py results --data_folder ./detections --output_results ./analysis
```

---

## 📋 Test Structure

| Directory | Purpose | Run Command |
|-----------|---------|------------|
| `tests/unit/` | Individual function tests | `pytest tests/unit/` |
| `tests/integration/` | Workflow tests | `pytest tests/integration/` |
| `tests/validation/` | Output correctness | `pytest tests/validation/` |

---

## 🔧 Common Commands

```bash
# Run ALL tests
pytest

# Run with coverage
pytest --cov=src --cov=scripts

# Run specific test file
pytest tests/unit/test_utils.py -v

# Run specific test
pytest tests/unit/test_utils.py::TestIOUCalculation::test_calc_iou_identical_boxes -v

# Stop on first failure
pytest -x

# Show print output
pytest -s
```

---

## 📚 Full Documentation

See **CLAUDE.md** for complete guide:
- Project structure
- Detailed algorithm explanations
- Development workflow
- CI/CD setup
- Troubleshooting

---

## ✅ Pre-commit Checks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

Built with ❤️ for whistle recognition. Questions? See CLAUDE.md.
