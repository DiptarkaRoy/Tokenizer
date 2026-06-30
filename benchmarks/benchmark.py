"""
benchmarks/benchmark.py
Performance evaluation suite comparing the custom BPE Tokenizer 
against OpenAI's tiktoken execution framework.
"""

import time
import tiktoken
from src.tokenizer import Tokenizer

# Test corpus incorporating code, text, and structural spacing variations
SAMPLE_DATA = """
def train_bpe_model(corpus: str, target_vocab: int) -> dict:
    \"\"\"
    Optimized Byte Pair Encoding processing array engine loop.
    Iterates over byte sequences to extract high-frequency merges.
    \"\"\"
    tokens = list(corpus.encode("utf-8"))
    while len(tokens) < target_vocab:
        pairs = get_stats(tokens)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        tokens = merge(tokens, best)
    return tokens

The quick brown fox jumps over the lazy dog multiple times.
Let's verify how contractions like won't, couldn't, and shouldn't are isolated.
Unicode consistency checking layer: ¡Hola! नमस्ते! ধন্যবাদ! 1234567890.
""" * 500  # Scale up the sample size for valid speed tracking metrics


def run_benchmark():
    print("=" * 70)
    print("     BPE TOKENIZER PERFORMANCE BENCHMARK SUITE     ")
    print("=" * 70)

    # 1. Initialize custom tokenizer and train it on the text block
    print("\n[1/3] Initializing and Training Custom Tokenizer...")
    custom_tokenizer = Tokenizer()
    
    # Train custom tokenizer with a 10,000 vocab size allocation target
    start_train = time.perf_counter()
    custom_tokenizer.train(vocab_size=10000, text=SAMPLE_DATA)
    train_duration = time.perf_counter() - start_train
    print(f"✓ Custom Tokenizer trained successfully in {train_duration:.4f} seconds.")

    # 2. Reference production framework configuration
    print("\n[2/3] Fetching Reference Model (tiktoken 'cl100k_base')...")
    reference_tokenizer = tiktoken.get_encoding("cl100k_base")

    # Calculate dataset dimensions
    raw_bytes_count = len(SAMPLE_DATA.encode("utf-8"))
    raw_megabytes = raw_bytes_count / (1024 * 1024)
    print(f"Dataset Size: {raw_bytes_count:,} bytes ({raw_megabytes:.3f} MB)")

    # 3. Processing performance checks
    print("\n[3/3] Executing Processing Pipeline Evaluations...")
    print("-" * 70)

    # Custom Tokenizer Performance Metric
    t0_custom = time.perf_counter()
    custom_tokens = custom_tokenizer.encode(SAMPLE_DATA)
    t1_custom = time.perf_counter()
    custom_duration = t1_custom - t0_custom
    custom_throughput = raw_megabytes / custom_duration

    # Reference Tokenizer Performance Metric
    t0_ref = time.perf_counter()
    ref_tokens = reference_tokenizer.encode(SAMPLE_DATA)
    t1_ref = time.perf_counter()
    ref_duration = t1_ref - t0_ref
    ref_throughput = raw_megabytes / ref_duration

    # Summary Output Table Display
    print(f"{'METRIC':<25} | {'CUSTOM BPE':<18} | {'TIKTOKEN (GPT-4)':<18}")
    print("-" * 70)
    print(f"{'Token Count (Lower=Best)':<25} | {len(custom_tokens):<18,} | {len(ref_tokens):<18,}")
    print(f"{'Compression Ratio':<25} | {raw_bytes_count / len(custom_tokens):<18.2f} | {raw_bytes_count / len(ref_tokens):<18.2f}")
    print(f"{'Encoding Time':<25} | {custom_duration:.5f}s{'' :<12} | {ref_duration:.5f}s")
    print(f"{'Throughput (Higher=Best)':<25} | {custom_throughput:.2f} MB/s{'' :<9} | {ref_throughput:.2f} MB/s")
    print("-" * 70)

    print("\n💡 Architectural Notes for GitHub README:")
    print("- Tiktoken utilizes a highly optimized compiled C++ core underlying engine.")
    print("- Tiktoken uses a much larger preset vocabulary (100k+), allowing for higher compression ratios.")


if __name__ == "__main__":
    run_benchmark()