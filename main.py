import os
import regex as re
from src.tokenizer import Tokenizer

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # 1. READ RESOURCE DATA
    # Load a large text corpus (e.g., from Project Gutenberg) to discover frequent character patterns
    file_path = "ProjectGutenberg.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Please place the text file in your workspace directory.")

    with open(file_path, 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    print(f"File loaded successfully.")
    print(f"Number of characters in training corpus: {len(sample_text)}")
    
    # 2. CORPUS PREPROCESSING & INITIAL VOCABULARY SETUP
    # GPT splitting rule regex: splits text into structural chunks (words, numbers, contractions, punctuation)
    pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    preprocessed_text = pattern.findall(sample_text)    
    
    # Stripping removes spaces, but spaces are vital structural context in BPE (e.g., ' the' vs 'the').
    preprocessed_text = [item for item in preprocessed_text if item]
    
    # Find all unique textual sub-chunks extracted by the regex
    words = sorted(set(preprocessed_text))
    
    # 3. INITIALIZE TOKENIZER & DEFINE MERGE LIMITS
    tokenizer = Tokenizer()
    
    # NOTE ON VOCAB_SIZE: 
    # Setting vocab_size to len(vocabulary) on a large book will create thousands of merges,
    # making pure Python extremely slow. For your initial test run, let's target 256 base bytes + 50 custom merges.
    # Change 'target_vocab_size' to len(vocabulary) only if you want a complete text extraction vocabulary.
    target_vocab_size = 256 + 50 
    
    # 4. TRAINING PHASE
    # Feed the text to the BPE algorithm. It transforms text to UTF-8 bytes and iteratively combines top pairs.
    print(f"\n--- Starting BPE Tokenizer Training (Target Vocab: {target_vocab_size}) ---")
    tokenizer.train(vocab_size=target_vocab_size, text=sample_text)

    # 5. TEST COMPRESSION ENGINE
    # Verify the code compiles and correctly processes unseen evaluations
    test_sentence = "Hello simple test."
    
    print("\n--- Phase 1: Encoding Evaluation ---")
    # Compressed string turns into custom trained numerical Token IDs
    encoded_tokens = tokenizer.encode(test_sentence)
    print(f"Original String Input:  '{test_sentence}'")
    print(f"Resulting Token IDs:    {encoded_tokens}")
    
    print("\n--- Phase 2: Decoding Evaluation ---")
    # Numerical IDs are mapped back using the byte decoder to reconstruct the original text string
    decoded_text = tokenizer.decode(encoded_tokens)
    print(f"Decoded String Output: '{decoded_text}'")
    
    # Verify integrity of output strings
    assert test_sentence == decoded_text, "Error: Tokenizer decompression did not match original input string!"
    print("\n[Success] Tokenization pipeline validated successfully!")