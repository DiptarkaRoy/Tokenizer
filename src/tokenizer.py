import regex as re

class Tokenizer:
    def __init__(self):
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.encoder={}
        self.decoder={}
        self.merges = {}  # Maps (id1, id2) -> new_id to track merge execution priority

    # Function to get adjacent pairs of IDs from a list of lists of IDs
    def get_adjacent_pairs_count(self, ids_list):
        adjacent_pairs={}
        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                adjacent_pairs[pair]=adjacent_pairs.get(pair, 0) + 1

        return adjacent_pairs

    def merge_adjacent_pairs(self, ids_list, adjacent_pairs, new_id):
        new_ids_list=[]
        for ids in ids_list:
            new_ids=[]
            i=0
            while i < len(ids):
                if i < len(ids)-1 and (ids[i], ids[i+1]) == adjacent_pairs:
                    new_ids.append(new_id)
                    i+=2
                else:
                    new_ids.append(ids[i])
                    i+=1
            new_ids_list.append(new_ids)

        return new_ids_list
    
    def train(self, vocab_size, text):        
        self.encoder = {bytes([i]): i for i in range(256)}
        self.merges = {}

        preprocessed_text = self.pattern.findall(text)
        preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
        words = sorted(set(preprocessed_text))
        ids_list=[list(chunks.encode('utf-8')) for chunks in words]

        current_idx = 256
        num_merges = vocab_size - 256  # Subtract the initial 256 byte values as utf-8 has 256 unique byte values.


        for k in range(num_merges):
            adjacent_pairs = self.get_adjacent_pairs_count(ids_list)
            if not adjacent_pairs:
                break
                
            most_frequent_pair = max(adjacent_pairs, key=adjacent_pairs.get)
            print(f"Iteration {k}: Most frequent pair: {most_frequent_pair}")

            # Keep track of this merge rule and its rank priority
            self.merges[most_frequent_pair] = current_idx

            # Correctly merge the sequence data
            ids_list = self.merge_adjacent_pairs(ids_list, most_frequent_pair, current_idx)

            # Retrieve raw bytes for both halves of the pair
            bytes_part1 = self._get_bytes_for_id(most_frequent_pair[0], self.encoder)
            bytes_part2 = self._get_bytes_for_id(most_frequent_pair[1], self.encoder)
            
            # FIX: Concatenate the raw byte sequences together
            byte_piece = bytes_part1 + bytes_part2
                    
            # Store the new mapping (Bytes -> New ID)
            self.encoder[byte_piece] = current_idx
            current_idx += 1
            
        # Rebuild the decoder at the very end of training
        self.decoder = {v: k for k, v in self.encoder.items()}
    
    def _get_bytes_for_id(self, idx, encoder_dict):
        # If it's a base byte, we can construct it instantly
        if idx < 256:
            return bytes([idx])
        
        # Otherwise, look it up dynamically from what we've built
        return next(k for k, v in encoder_dict.items() if v == idx)
    
    def encode(self, text):
        """
        Encodes raw incoming string text into token IDs based on learned BPE merges.
        """
        text_chunks = self.pattern.findall(text)
        final_ids = []

        for chunk in text_chunks:
            # Convert text chunk into raw utf-8 byte values
            chunk_ids = list(chunk.encode('utf-8'))

            while len(chunk_ids) >= 2:
                # Find all current pairs inside this isolated text chunk
                adjacent_pairs = {}
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + 1
                
                # Out of all available pairs here, select the one that was learned EARLIEST in training.
                # If a pair wasn't learned during training, give it an infinite rank (ignored).
                best_pair = min(adjacent_pairs, key=lambda p: self.merges.get(p, float('inf')))
                
                # If the best pair is not a valid trained merge, we can no longer compress this chunk
                if best_pair not in self.merges:
                    break
                
                # Execute the single merge across the chunk sequence
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
                chunk_ids = new_ids

            final_ids.extend(chunk_ids)
            
        return final_ids
    
    def decode(self, ids):
        """
        Decodes token IDs back into raw readable string text.
        """
        byte_tokens = []
        for idx in ids:
            if idx in self.decoder:
                byte_tokens.append(self.decoder[idx])
            else:
                byte_tokens.append(bytes([idx]))
                
        # Combine all parts together and decode into raw UTF-8 string safely
        return b"".join(byte_tokens).decode("utf-8", errors="replace")