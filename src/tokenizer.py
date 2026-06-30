"""
src/tokeniser.py
Stateful Tokenizer wrapper class for managing the BPE encoding, decoding, 
training life cycle, and structural special token parsing.
"""

import regex as re
from src.core import (
    compute_adjacent_pairs_frequency,
    merge_sequence_pairs,
    serialize_bpe_model,
    deserialize_bpe_model
)


class Tokenizer:
    def __init__(self):
        # OpenAI's GPT-2 / GPT-4 regex pattern split rules to control token splitting boundaries
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Core vocabulary structures
        self.encoder = {}         # Maps raw Bytes -> Integer Token ID
        self.decoder = {}         # Maps Integer Token ID -> raw Bytes
        self.merges = {}          # Maps (id1, id2) -> new_id to track merge execution order
        self.special_tokens = {}  # Maps string special tokens (e.g. "<|endoftext|>") -> ID
        self.inverse_special_tokens = {} # Maps ID -> string special token

        # Seed the base vocabulary with standard 256 physical byte values
        self._initialize_base_vocabulary()

    def _initialize_base_vocabulary(self):
        """Initializes vocabulary maps with the foundational 256 individual byte combinations."""
        self.encoder = {bytes([i]): i for i in range(256)}
        self.decoder = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def _rebuild_vocabulary_maps(self):
        """Reconstructs the full encoder and decoder maps dynamically from learned merge rules."""
        # Reset to base 256 bytes
        self.encoder = {bytes([i]): i for i in range(256)}
        
        # Sort merges by their target new_id to recreate them in exact chronological order
        sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])
        
        for (id1, id2), new_id in sorted_merges:
            bytes_part1 = self._get_bytes_for_id(id1)
            bytes_part2 = self._get_bytes_for_id(id2)
            self.encoder[bytes_part1 + bytes_part2] = new_id
            
        # Add special tokens to encoder
        for token_str, token_id in self.special_tokens.items():
            self.encoder[token_str.encode('utf-8')] = token_id

        # Generate inverse decoder maps
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def _get_bytes_for_id(self, idx: int) -> bytes:
        """Helper to resolve or recover the exact raw bytes representation of a given Token ID."""
        if idx < 256:
            return bytes([idx])
        return next(k for k, v in self.encoder.items() if v == idx)

    def register_special_tokens(self, special_tokens_dict: dict[str, int]):
        """
        Registers custom special structural tokens into the tokenizer vocabulary.
        Example: tokenizer.register_special_tokens({"<|endoftext|>": 100001})
        """
        for token_str, token_id in special_tokens_dict.items():
            if token_id < 256:
                raise ValueError(f"Token ID {token_id} conflicts with standard foundational 256 byte keys.")
            self.special_tokens[token_str] = token_id
        self._rebuild_vocabulary_maps()

    def train(self, vocab_size: int, text: str):
        """Trains the tokenizer by iteratively merging the most frequent token sequences."""
        if vocab_size <= 256:
            raise ValueError("Target vocabulary size must be greater than the base 256 bytes.")

        self._initialize_base_vocabulary()
        
        # Split text into pieces using the structural GPT regex
        preprocessed_text = self.pattern.findall(text)
        preprocessed_text = [item for item in preprocessed_text if item]
        
        # Get unique words/chunks for processing footprint optimization
        words = sorted(set(preprocessed_text))
        ids_list = [list(chunks.encode('utf-8')) for chunks in words]

        current_idx = 256
        num_merges = vocab_size - 256

        for k in range(num_merges):
            adjacent_pairs = compute_adjacent_pairs_frequency(ids_list)
            if not adjacent_pairs:
                break
                
            most_frequent_pair = max(adjacent_pairs, key=adjacent_pairs.get)
            
            # Register the merge priority rule
            self.merges[most_frequent_pair] = current_idx

            # Execute sequence compression across the corpus
            ids_list = merge_sequence_pairs(ids_list, most_frequent_pair, current_idx)
            current_idx += 1
            
        # Materialize the final encoding/decoding hash tables
        self._rebuild_vocabulary_maps()

    def encode(self, text: str, allowed_special: set[str] = None) -> list[int]:
        """Encodes string text into a compressed list of unique token IDs."""
        if allowed_special is None:
            allowed_special = set()

        # Handle special tokens via split if strings are explicitly allowed
        if allowed_special and self.special_tokens:
            # Create regex pattern to match any allowed special tokens
            special_pattern = re.compile("(" + "|".join(re.escape(t) for t in allowed_special if t in self.special_tokens) + ")")
            parts = special_pattern.split(text)
        else:
            parts = [text]

        final_ids = []
        for part in parts:
            # If this segment is an allowed special token, process it directly
            if part in allowed_special and part in self.special_tokens:
                final_ids.append(self.special_tokens[part])
                continue
                
            # Otherwise, run standard BPE compression across the text chunk
            text_chunks = self.pattern.findall(part)
            for chunk in text_chunks:
                chunk_ids = list(chunk.encode('utf-8'))

                while len(chunk_ids) >= 2:
                    adjacent_pairs = {}
                    for pair in zip(chunk_ids, chunk_ids[1:]):
                        adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + 1
                    
                    # Identify the pair that was discovered EARLIEST in the training history
                    best_pair = min(adjacent_pairs, key=lambda p: self.merges.get(p, float('inf')))
                    
                    if best_pair not in self.merges:
                        break  # No more valid learned rules apply to this sequence
                    
                    new_id = self.merges[best_pair]
                    
                    # Merge loop execution
                    new_ids = []
                    i = 0
                    while i < len(chunk_ids):
                        if i < len(chunk_ids) - 1 and (chunk_ids[i], chunk_ids[i+1]) == best_pair:
                            new_ids.append(new_id)
                            i += 2
                        else:
                            new_ids.append(chunk_ids[i])
                            i += 1
                    chunk_ids = new_ids

                final_ids.extend(chunk_ids)
                
        return final_ids

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs back into a safe, human-readable UTF-8 string."""
        byte_tokens = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                # Materialize the special token text back safely
                byte_tokens.append(self.inverse_special_tokens[idx].encode('utf-8'))
            elif idx in self.decoder:
                byte_tokens.append(self.decoder[idx])
            else:
                byte_tokens.append(bytes([idx]))
                
        return b"".join(byte_tokens).decode("utf-8", errors="replace")

    def save(self, file_path: str):
        """Persists the tokenizer merge patterns and state configs onto disk."""
        serialize_bpe_model(file_path, self.merges, self.special_tokens)

    def load(self, file_path: str):
        """Loads a model file state from disk and hot-reconstructs memory lookups."""
        self.merges, self.special_tokens = deserialize_bpe_model(file_path)
        self._rebuild_vocabulary_maps()