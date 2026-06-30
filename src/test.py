from src.tokenizer import Tokenizer

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Define a sample text corpus to train our tokenizer on
    training_corpus = "Hello world! This is a simple tokenizer test to verify GPT BPE implementation rules."

    # 2. Initialize the Tokenizer
    tokenizer = Tokenizer()

    # 3. Train the Tokenizer (256 base bytes + 5 custom merges = 261 vocab size)
    tokenizer.train(vocab_size=261, text=training_corpus)

    # 4. Test on a completely new sentence
    test_sentence = "Hello simple test."
    
    print("--- Encoding ---")
    encoded_tokens = tokenizer.encode(test_sentence)
    print(f"Original Text: {test_sentence}")
    print(f"Token IDs:     {encoded_tokens}")
    
    print("\n--- Decoding ---")
    decoded_text = tokenizer.decode(encoded_tokens)
    print(f"Decoded Text:  {decoded_text}")