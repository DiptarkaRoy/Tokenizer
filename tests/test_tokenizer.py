"""
tests/test_tokenizer.py
Unit test suite utilizing pytest to validate BPE tokenizer invariants, 
edge cases, special tokens, and file system serialization.
"""

import os
import pytest
from src.tokenizer import Tokenizer


@pytest.fixture
def trained_tokenizer():
    """Provides a basic pre-trained tokenizer instance for standard execution tests."""
    tokenizer = Tokenizer()
    sample_corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Tokenization is an essential step in modern deep learning models. "
        "Let's test contractions like don't, I'm, and structural spaces!"
    )
    # Train with a compact vocabulary target size of 300 (256 base + 44 merges)
    tokenizer.train(vocab_size=300, text=sample_corpus)
    return tokenizer


def test_identity_invariant_roundtrip(trained_tokenizer):
    """Verifies that any arbitrary text sequence can be encoded and decoded perfectly without data loss."""
    test_strings = [
        "Hello World!",
        "Let's check how contractions work under the hood.",
        "Testing numbers 1234567890 and punctuation symbols: @#$^&*()_+.",
        "   Multiple    consecutive   spaces   should   survive   intact.",
        "Handling non-English unicode characters: ¡Hola! Bonjour! नमस्ते! ধন্যবাদ!"
    ]
    
    for text in test_strings:
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text, f"Roundtrip failed for: {text!r}"


def test_base_vocabulary_bounds():
    """Ensures that a brand new tokenizer instance correctly registers exactly the base 256 physical bytes."""
    tokenizer = Tokenizer()
    assert len(tokenizer.encoder) == 256
    assert len(tokenizer.decoder) == 256
    assert max(tokenizer.decoder.keys()) == 255


def test_special_tokens_handling(trained_tokenizer):
    """Validates that registered special tokens remain isolated blocks and do not get split up by regular BPE rules."""
    special_map = {"<|endoftext|>": 500, "<|system_prompt|>": 501}
    trained_tokenizer.register_special_tokens(special_map)
    
    input_text = "<|system_prompt|> System initializing... <|endoftext|>"
    
    # Test encoding with special tokens activated
    allowed = {"<|endoftext|>", "<|system_prompt|>"}
    encoded = trained_tokenizer.encode(input_text, allowed_special=allowed)
    
    # Ensure the exact IDs specified were injected into the token array
    assert 501 in encoded
    assert 500 in encoded
    
    # Test roundtrip recovery
    decoded = trained_tokenizer.decode(encoded)
    assert decoded == input_text


def test_special_tokens_ignored_when_not_allowed(trained_tokenizer):
    """Verifies that if a special token string is present but not explicitly allowed, it is treated as regular text."""
    special_map = {"<|endoftext|>": 500}
    trained_tokenizer.register_special_tokens(special_map)
    
    input_text = "Keep it raw: <|endoftext|>"
    
    # Explicitly do NOT allow the special token during encoding
    encoded = trained_tokenizer.encode(input_text, allowed_special=set())
    
    # The dedicated special token ID should NOT appear because it should be treated as separate characters
    assert 500 not in encoded


def test_model_serialization_lifecycle(trained_tokenizer, tmp_path):
    """Validates that a trained model state configuration can be saved to disk and loaded back with identical maps."""
    # Use pytest's secure temporary directory fixture to store files safely
    model_file = os.path.join(tmp_path, "tokenizer.bpe")
    
    # Register a sample special token to check config inclusion
    trained_tokenizer.register_special_tokens({"<|pad|>": 400})
    
    # Save the current state config
    trained_tokenizer.save(model_file)
    assert os.path.exists(model_file)
    
    # Spin up a brand new fresh tokenizer instance and load the configuration state
    new_tokenizer = Tokenizer()
    new_tokenizer.load(model_file)
    
    # Assert structural alignment between both memory configurations
    assert new_tokenizer.merges == trained_tokenizer.merges
    assert new_tokenizer.special_tokens == trained_tokenizer.special_tokens
    assert new_tokenizer.encoder == trained_tokenizer.encoder
    assert new_tokenizer.decoder == trained_tokenizer.decoder
    
    # Verify functional parity via cross-instance test
    sample_text = "Testing cross-instance encoding mechanics!"
    assert new_tokenizer.encode(sample_text) == trained_tokenizer.encode(sample_text)