"""
Unit tests for machine learning data preprocessing and postprocessing.
"""
from ml.model_def import preprocess, tokenize, postprocess


def test_preprocess():
    """Ensure punctuation is removed and spacing is normalized."""
    assert preprocess("Hello?! World;") == "Hello World"


def test_tokenize():
    """Ensure text is correctly split into tokens."""
    assert tokenize("Hello world") == ["Hello", "world"]


def test_postprocess():
    """Ensure special tokens are removed and punctuation is correctly attached."""
    assert postprocess(["Hello", "world", ".", "<unk>"]) == "Hello world."