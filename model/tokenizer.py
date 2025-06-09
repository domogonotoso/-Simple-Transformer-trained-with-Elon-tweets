from transformers import GPT2Tokenizer
import os
from typing import List

class MyTokenizer:
    def __init__(self, model_name="gpt2", cache_dir="./tokenizer"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Make sure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)            
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
