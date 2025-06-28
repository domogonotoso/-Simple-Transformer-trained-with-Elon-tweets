import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import yaml
import os
from transformers import GPT2Tokenizer

from utils.plot import plot_loss
from model.transformer import GPTMini
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    x_list, y_list = zip(*batch)  
    x_padded = pad_sequence(x_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    return x_padded, y_padded


# Load hyperparameters
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config['batch_size']
BLOCK_SIZE = config['block_size']
EPOCHS = config['epochs']
LR = float(config['learning_rate'])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

# Dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            tokens = tokenizer.encode(line.strip())
            if len(tokens) > 1:
                x = tokens[:-1]
                y = tokens[1:]
                self.data.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Train loop
def train():
    dataset = TextDataset('data/cleaned.txt', tokenizer)

    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty. Check block_size or tokenization.")

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    model = GPTMini(vocab_size=tokenizer.vocab_size).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    train_losses = []

    for epoch in range(EPOCHS):
        total_loss = 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"checkpoints/gptmini_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), "checkpoints/gptmini_final.pth")
    print("✅ Final model saved.")

    plot_loss(train_losses, save_path="results/loss_plot.png")


if __name__ == "__main__":
    train()
