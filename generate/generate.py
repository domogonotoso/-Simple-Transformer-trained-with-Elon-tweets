import torch
import os
from transformers import GPT2Tokenizer
from model.transformer import GPTMini
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Hyperparameters
TEMPERATURE = cfg.get("temperature", 0.3)
TOP_K = cfg.get("top_k", 5)
MAX_NEW_TOKENS = cfg.get("max_new_tokens", 100)
MODEL_PATH = "checkpoints/gptmini_final.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # for safety

def generate_text(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE, top_k: int = TOP_K):
    # Load model
    model = GPTMini(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

    # Autoregressive generation
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[0, -1, :] / temperature

        if top_k is not None:
            top_k = min(top_k, next_token_logits.size(-1))
            values, indices = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits[indices] = values

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output

if __name__ == "__main__":
    prompt = input("💬 Enter a prompt: ")
    output = generate_text(prompt)
    print("\n📝 Generated text:\n")
    print(output)

    os.makedirs("results", exist_ok=True)
    with open("results/samples.txt", "a", encoding="utf-8") as f:
        f.write("PROMPT:\n" + prompt + "\n\n")
        f.write("GENERATED:\n" + output + "\n")
        f.write("=" * 50 + "\n")
