# Test Coverage and Execution Guide

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Specific test file
pytest tests/unit/test_transcription.py

# Specific test class or function
pytest tests/unit/test_classification.py::TestClassifier::test_init_with_api_url_parameter
```

### Run with Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=modules --cov=database --cov=storage --cov=pipeline --cov-report=html

# View coverage in terminal
pytest --cov=modules --cov=database --cov=storage --cov=pipeline --cov-report=term-missing

# Open HTML report (coverage report will be in htmlcov/index.html)
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Tests with Markers
```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run tests that don't require GPU
pytest -m "not requires_gpu"
```

### Verbose Output
```bash
# More detailed output
pytest -v

# Very verbose with full diff
pytest -vv

# Show print statements
pytest -s
```

### Run Tests in Parallel
```bash
# Install pytest-xdist first: pip install pytest-xdist
pytest -n auto  # Auto-detect number of CPUs
pytest -n 4     # Use 4 workers
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (test individual components)
│   ├── test_transcription.py  # Tests for transcription module
│   ├── test_classification.py # Tests for classification module
│   ├── test_frequency.py      # Tests for frequency analysis
│   ├── test_compliance.py     # Tests for compliance verification
│   ├── test_storage.py        # Tests for MinIO storage client
│   └── test_database.py       # Tests for database operations
```

## Writing New Tests

### Example Unit Test
```python
import pytest
from mymodule import MyClass

class TestMyClass:
    """Test cases for MyClass."""
    
    def test_initialization(self):
        """Test that MyClass initializes correctly."""
        obj = MyClass(param="value")
        assert obj.param == "value"
    
    def test_method_with_mock(self, mocker):
        """Test method with mocked dependency."""
        mock_api = mocker.patch('mymodule.external_api')
        mock_api.return_value = "mocked response"
        
        obj = MyClass()
        result = obj.method_that_uses_api()
        
        assert result == "expected result"
        mock_api.assert_called_once()
```

### Using Fixtures
```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using fixture."""
    assert sample_data["key"] == "value"
```

### Marking Tests
```python
@pytest.mark.unit
def test_simple_function():
    """Simple unit test."""
    assert 1 + 1 == 2

@pytest.mark.slow
def test_long_running_process():
    """Test that takes significant time."""
    # Long test code here
    pass
```

## Available Fixtures

See `conftest.py` for all available fixtures:

- `temp_dir` - Temporary directory for test files
- `sample_config` - Sample configuration dictionary
- `config_file` - Temporary config YAML file
- `sample_audio_data` - Generated audio data (numpy array)
- `sample_audio_segment` - AudioSegment instance
- `sample_audio_file` - Audio WAV file on disk
- `sample_transcription_result` - TranscriptionResult instance
- `sample_classification_result` - ClassificationResult instance
- `mock_minio_client` - Mocked MinIO client
- `in_memory_db` - In-memory SQLite database
- `sample_manifest` - Manifest database record
- `sample_manifest_call` - ManifestCall database record
- `mock_transcriber` - Mocked Transcriber
- `mock_classifier` - Mocked Classifier

## Troubleshooting

### Tests fail due to missing dependencies
```bash
pip install -r requirements.txt
```

### Import errors
```bash
# Make sure you're in the project root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### GPU-related failures
```bash
# Run without GPU tests
pytest -m "not requires_gpu"
```

### Database connection issues
```bash
# Tests use in-memory SQLite by default
# Check conftest.py fixture configuration
```
