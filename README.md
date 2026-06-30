# Custom Byte-Level BPE Tokenizer

A from-scratch Python implementation of a Byte-Pair Encoding (BPE) tokenizer matching the architectural design choices of OpenAI's GPT models. This engine converts raw text sequences into compressed byte-level token IDs, seamlessly handles custom special tokens, and prevents cross-boundary merging using advanced Unicode-aware regular expressions.

## Features
* **Byte-Level Core:** Initialized with the foundational 256 structural byte keys to flawlessly handle Out-Of-Vocabulary (OOV) tokens without throwing errors.
* **GPT Splitting Rules:** Employs the native GPT regex split to preserve contractions, punctuation spacing, and word clusters.
* **Special Token Injection:** Supports runtime isolation of custom tokens (e.g., `<|system_prompt|>`, `<|endoftext|>`).
* **Safe Serialization:** Rejects insecure pickles; saves/loads configurations cleanly via flat JSON translation files.

## Project Architecture

Organized using standard open-source library structures:

```text
tokenizer-repo/
├── src/
│   ├── __init__.py
│   ├── core.py          # Vectorized algorithms & file serialization utilities
│   └── tokeniser.py     # Stateful user-facing Tokenizer engine class
├── tests/
│   └── test_tokenizer.py# PyTest suite isolating roundtrip invariants
├── benchmarks/
│   └── benchmark.py     # Throughput & compression benchmarking suite
├── main.py              # End-to-end literature training playground
├── pyproject.toml       # Modern PEP 621 package build configuration
├── requirements.txt     # Python dependency version anchors
└── README.md            # Repository documentation showcase
```

## Installation & Setup

1. Clone the repository and initialize your environment.
2. Install the library in editable development mode:
```bash
pip install -e .
```
## Testing

The repository features an automated test suite driven by `pytest` to enforce tokenization invariants, protect against processing regressions, and validate data persistence mechanics. Execute the test runner from the root directory of your project: 

```bash
pytest -v
```

## Performance Benchmarks
Run python `benchmarks/benchmark.py` to evaluate compression ratios and throughput limits:

```bash
python benchmarks/benchmark.py
```