import torch
import os
from model.tokenizer import MyTokenizer
from model.transformer import GPTMini

# Hyperparameters
MODEL_PATH = "checkpoints/best_gptmini.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_NEW_TOKENS = 100  # maximum number of tokens to generate
BLOCK_SIZE = 128

def generate_text(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 1.0, top_k: int = 50):
    # Load tokenizer and model
    tokenizer = MyTokenizer()
    model = GPTMini(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[-BLOCK_SIZE:]  # cut to block size
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    # Autoregressive generation loop
    for _ in range(max_new_tokens):
        input_crop = input_tensor[:, -BLOCK_SIZE:]  # ensure fixed block size
        with torch.no_grad():
            logits = model(input_crop)  # (1, T, vocab_size)
        next_token_logits = logits[0, -1, :] / temperature  # (vocab_size,)

        # Optional: top-k sampling
        if top_k is not None:
            top_k = min(top_k, next_token_logits.size(-1))
            values, indices = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits[indices] = values

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

    # Decode
    generated_ids = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


if __name__ == "__main__":
    prompt = input("💬 Enter a prompt: ")
    output = generate_text(prompt)
    print("\n📝 Generated text:\n")
    print(output)

    # Save to file
    os.makedirs("results", exist_ok=True)
    with open("results/samples.txt", "a", encoding="utf-8") as f:
        f.write("PROMPT:\n" + prompt + "\n\n")
        f.write("GENERATED:\n" + output + "\n")
        f.write("=" * 50 + "\n")
