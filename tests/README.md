# WRS Test Suite

Complete test coverage for Whistles Recognition System (WRS).

## Test Organization

```
tests/
├── conftest.py               # Shared fixtures for all tests
├── unit/                     # Unit tests (individual functions)
│   ├── test_utils.py         # src/utils.py functions
│   ├── test_wrsapplication.py # WRSapplication.py module
│   └── test_wrsresults.py    # WRSresults.py module
├── integration/              # Integration tests (workflows)
│   └── test_pipeline.py      # End-to-end pipelines
└── validation/               # Output validation
    └── test_results.py       # Results format & correctness
```

## Running Tests

### All Tests
```bash
pytest -v
```

### By Category
```bash
pytest tests/unit/ -v              # Only unit tests
pytest tests/integration/ -v        # Only integration tests
pytest tests/validation/ -v         # Only validation tests
```

### With Coverage
```bash
pytest --cov=src --cov=scripts --cov-report=term-missing
pytest --cov=src --cov=scripts --cov-report=html  # HTML report
```

### Specific Test
```bash
pytest tests/unit/test_utils.py::TestIOUCalculation::test_calc_iou_identical_boxes -v
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual functions in isolation.

- `test_utils.py`:
  - Box merging algorithms
  - IoU calculations
  - File discovery
  - Configuration loading
  - Histogram generation

- `test_wrsapplication.py`:
  - File listing for images
  - Output directory handling
  - JSON structure validation
  - Merge threshold effects

- `test_wrsresults.py`:
  - JSON parsing
  - Parameter extraction
  - DataFrame construction
  - CSV formatting

### Integration Tests (`tests/integration/`)

**Purpose**: Test complete workflows connecting multiple components.

- `test_pipeline.py`:
  - JSON → CSV processing pipeline
  - Box merging + analysis chain
  - Histogram generation workflow
  - Multi-file aggregation
  - Data format conversions

### Validation Tests (`tests/validation/`)

**Purpose**: Verify output correctness and format compliance.

- `test_results.py`:
  - Detection format validation
  - Merged box consistency
  - Physical parameter validity
  - Data integrity through pipeline
  - Metadata preservation

## Test Fixtures

Common fixtures available in `conftest.py`:

```python
@pytest.fixture
def temp_dir()                # Temporary directory
def sample_bbox()             # Sample bounding box
def sample_detection()        # Sample detection dict
def overlapping_detections()  # Multiple overlapping boxes
def contained_detections()    # Containment test case
def config_yaml_path()        # Test config file
```

## Coverage Goals

| Component | Target | Status |
|-----------|--------|--------|
| `src/utils.py` | 95% | ✅ |
| `scripts/WRSapplication.py` | 80% | ✅ |
| `scripts/WRSresults.py` | 80% | ✅ |
| **Overall** | **90%** | 🔄 |

## Continuous Integration

Tests run automatically on:
- **Push** to `main` or `develop`
- **Pull requests** to `main` or `develop`

See `.github/workflows/ci.yml` for details.

## Debugging Failed Tests

```bash
# Verbose output
pytest tests/unit/test_utils.py -vv

# Stop on first failure
pytest tests/unit/test_utils.py -x

# Show print statements
pytest tests/unit/test_utils.py -s

# Drop into debugger on failure
pytest tests/unit/test_utils.py --pdb

# Show local variables on failure
pytest tests/unit/test_utils.py -l
```

## Adding New Tests

1. Create test file in appropriate directory
2. Name it `test_*.py`
3. Create test classes: `class Test*`
4. Create test functions: `def test_*`
5. Use fixtures from `conftest.py`
6. Run: `pytest tests/unit/test_new.py -v`

Example:
```python
import pytest
from module import my_function

class TestMyFunction:
    def test_basic(self):
        result = my_function(5)
        assert result == 10

    def test_with_fixture(self, sample_detection):
        assert sample_detection["class"] == "w"
```

## Performance Notes

- Unit tests: ~1 second
- Integration tests: ~2 seconds
- Validation tests: ~1 second
- **Total suite**: ~4-5 seconds

## Known Issues

None currently.

## Contributing

When contributing:
1. ✅ Add tests for new features
2. ✅ Ensure all tests pass: `pytest`
3. ✅ Check coverage: `pytest --cov=src`
4. ✅ Update this README if needed

---

**Last Updated**: June 16, 2026
