"""
Unit tests for the Deepo Translation System logic.
Validates text processing, tokenization, and deduplication filters.
"""
import pytest
import re
from ml.model_def import tokenize, preprocess

def test_deduplication_regex():
    """Unit test: Verifies the re.sub logic removes consecutive duplicate words."""
    # This matches the regex added to backend/app/main.py
    regex = r'\b(\w+)( \1\b)+'
    text = "¿Cuál es tu color favorito favorito favorito?"
    cleaned = re.sub(regex, r'\1', text)
    assert cleaned == "¿Cuál es tu color favorito?"

def test_tokenization_cjk():
    """Unit test: Ensures CJK characters are tokenized character-by-character."""
    text = "我爱你"  # INFO : Chinese "I love you"
    tokens = tokenize(text)
    assert tokens == ["我", "爱", "你"]

def test_text_preprocessing_cleanup():
    """Unit test: Ensures punctuation is handled and spacing is normalized."""
    text = "  Hello!!  World;  "
    cleaned = preprocess(text)
    assert cleaned == "Hello World"