# tests/test_predict.py
from src.utils import validate_input
import pytest

def test_validate_single_sample():
    sample = [5.1, 3.5, 1.4, 0.2]
    arr = validate_input(sample)
    assert arr.shape == (1, 4)

def test_invalid_input():
    with pytest.raises(ValueError):
        validate_input([1,2,3])  # wrong length
