"""
src/core.py
Core functional mathematical and processing engine for the Byte-Pair Encoding (BPE) algorithm.
Contains optimized helpers for tracking frequencies, sequence merging, and vocabulary storage.
"""

import json
from collections import Counter


def compute_adjacent_pairs_frequency(ids_list: list[list[int]]) -> dict[tuple[int, int], int]:
    """
    Optimized scanning of 2D token arrays to compute frequencies of consecutive pairs.
    Uses collections.Counter (C-optimized under the hood) for high throughput.
    """
    stats = Counter()
    for ids in ids_list:
        if len(ids) < 2:
            continue
        # zip() creates a memory-efficient iterator over adjacent pairs
        stats.update(zip(ids, ids[1:]))
    return dict(stats)


def merge_sequence_pairs(ids_list: list[list[int]], target_pair: tuple[int, int], new_id: int) -> list[list[int]]:
    """
    Iterates through a token matrix and compresses all instances of target_pair into new_id.
    Optimized using standard flat loop arrays.
    """
    new_ids_list = []
    p0, p1 = target_pair
    
    for ids in ids_list:
        if len(ids) < 2:
            new_ids_list.append(ids)
            continue
            
        new_ids = []
        i = 0
        while i < len(ids):
            # Check if the sliding window matches our highest priority pair
            if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                new_ids.append(new_id)
                i += 2  # Compress the pair, advance pointer past both tokens
            else:
                new_ids.append(ids[i])
                i += 1
        new_ids_list.append(new_ids)
        
    return new_ids_list


def serialize_bpe_model(file_path: str, merges: dict[tuple[int, int], int], special_tokens: dict[str, int] = None):
    """
    Saves the learned merge rules and special tokens into a standardized, human-readable file format.
    Does not use insecure Python pickles; instead uses an explicit structured dictionary config.
    """
    # Convert tuple keys (id1, id2) into readable strings "id1,id2" for valid JSON storage
    serializable_merges = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
    
    model_payload = {
        "model_type": "byte-level-bpe",
        "special_tokens": special_tokens or {},
        "merges": serializable_merges
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)


def deserialize_bpe_model(file_path: str) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    """
    Loads a saved BPE model configuration and reconstructs the memory structures.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        model_payload = json.load(f)
        
    raw_merges = model_payload.get("merges", {})
    special_tokens = model_payload.get("special_tokens", {})
    
    # Reconstruct stringified keys back into tuple[int, int]
    merges = {}
    for k, v in raw_merges.items():
        id1, id2 = map(int, k.split(","))
        merges[(id1, id2)] = v
        
    return merges, special_tokens