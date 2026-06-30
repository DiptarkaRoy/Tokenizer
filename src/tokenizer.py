import regex as re

class Tokenizer:
    def __init__(self):
        # OpenAI's GPT-2 regex pattern to split text into distinct semantic blocks
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.encoder = {}  # Maps raw Bytes -> Integer Token ID
        self.decoder = {}  # Maps Integer Token ID -> raw Bytes
        self.merges = {}   # Maps (id1, id2) -> new_id to keep track of training priority ranks

    def get_adjacent_pairs_count(self, ids_list):
        """Scans sequences of token IDs and returns frequencies of consecutive pairs."""
        adjacent_pairs = {}
        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + 1
        return adjacent_pairs

    def merge_adjacent_pairs(self, ids_list, target_pair, new_id):
        """Replaces all occurrences of target_pair with a single new_id across the corpus."""
        new_ids_list = []
        for ids in ids_list:
            new_ids = []
            i = 0
            while i < len(ids):
                # If we encounter the target pair, compress it into the new ID
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == target_pair:
                    new_ids.append(new_id)
                    i += 2  # Skip both elements
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids_list.append(new_ids)
        return new_ids_list
    
    def train(self, vocab_size, text):        
        """Trains the tokenizer by iteratively merging the most frequent byte/token pairs."""
        # Initialize the base vocabulary with the standard 256 structural byte options
        self.encoder = {bytes([i]): i for i in range(256)}
        self.merges = {}

        # Chunk the text using the structural GPT regex rules
        preprocessed_text = self.pattern.findall(text)
        
        # NOTE: We avoid .strip() here because spaces are foundational structures in GPT BPE
        preprocessed_text = [item for item in preprocessed_text if item]
        words = sorted(set(preprocessed_text))
        
        # Convert character strings into lists of raw byte integers
        ids_list = [list(chunks.encode('utf-8')) for chunks in words]

        current_idx = 256
        num_merges = vocab_size - 256

        for k in range(num_merges):
            adjacent_pairs = self.get_adjacent_pairs_count(ids_list)
            if not adjacent_pairs:
                break
                
            most_frequent_pair = max(adjacent_pairs, key=adjacent_pairs.get)
            print(f"Iteration {k}: Most frequent pair: {most_frequent_pair} -> Merged ID: {current_idx}")

            # Register the merge priority rule
            self.merges[most_frequent_pair] = current_idx

            # Update the numerical IDs list with our new merged token
            ids_list = self.merge_adjacent_pairs(ids_list, most_frequent_pair, current_idx)

            # Look up raw byte segments for both components to construct the merged token's bytes
            bytes_part1 = self._get_bytes_for_id(most_frequent_pair[0], self.encoder)
            bytes_part2 = self._get_bytes_for_id(most_frequent_pair[1], self.encoder)
            byte_piece = bytes_part1 + bytes_part2
                    
            # Update the encoder vocabulary map
            self.encoder[byte_piece] = current_idx
            current_idx += 1
            
        # Generate the inverse lookup table for decoding
        self.decoder = {v: k for k, v in self.encoder.items()}
    
    def _get_bytes_for_id(self, idx, encoder_dict):
        """Helper to safely fetch or reconstruct raw byte values for any vocabulary ID."""
        if idx < 256:
            return bytes([idx])
        return next(k for k, v in encoder_dict.items() if v == idx)
    
    def encode(self, text):
        """Encodes raw incoming string text into token IDs based on training merge rules."""
        text_chunks = self.pattern.findall(text)
        final_ids = []

        for chunk in text_chunks:
            chunk_ids = list(chunk.encode('utf-8'))

            while len(chunk_ids) >= 2:
                adjacent_pairs = {}
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + 1
                
                # Prioritize merging pairs that were discovered EARLIEST during training
                best_pair = min(adjacent_pairs, key=lambda p: self.merges.get(p, float('inf')))
                
                # If the best pair was never registered during training, we cannot compress further
                if best_pair not in self.merges:
                    break
                
                new_id = self.merges[best_pair]
                new_ids = []
                i = 0
                while i < len(chunk_ids):
                    if i < len(chunk_ids) - 1 and (chunk_ids[i], chunk_ids[i+1]) == best_pair:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(chunk_ids[i])
                        i += 1
                
                # FIX: We only reassign chunk_ids AFTER the inner loop completes fully!
                chunk_ids = new_ids

            final_ids.extend(chunk_ids)
            
        return final_ids
    
    def decode(self, ids):
        """Decodes token IDs back into a human-readable UTF-8 string."""
        byte_tokens = []
        for idx in ids:
            if idx in self.decoder:
                byte_tokens.append(self.decoder[idx])
            else:
                byte_tokens.append(bytes([idx]))
                
        return b"".join(byte_tokens).decode("utf-8", errors="replace")