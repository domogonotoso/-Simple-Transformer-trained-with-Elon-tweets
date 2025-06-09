import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import yaml
import os

from utils.plot import plot_loss
from model.tokenizer import MyTokenizer
from model.transformer import GPTMini

# Load hyperparameters from config.yaml
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config['batch_size']
BLOCK_SIZE = config['block_size']
EPOCHS = config['epochs']
LR = float(config['learning_rate'])
VAL_SPLIT = config.get('val_split', 0.1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Dataset class for tokenized tweets
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            tokens = tokenizer.encode(line.strip())
            for i in range(0, len(tokens) - BLOCK_SIZE):
                x = tokens[i:i + BLOCK_SIZE]
                y = tokens[i + 1:i + 1 + BLOCK_SIZE]
                self.data.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Evaluate model on validation set
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)


# Training loop
def train():
    tokenizer = MyTokenizer()
    dataset = TextDataset('data/processed.txt', tokenizer)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = GPTMini(vocab_size=tokenizer.vocab_size).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

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
            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_gptmini.pth")
            print("💾 Best model saved.")

    # Final model save
    torch.save(model.state_dict(), "checkpoints/last_gptmini.pth")
    print("✅ Final model saved.")

    # Plot training and validation loss
    plot_loss(train_losses, val_losses, save_path="results/loss_plot.png")


if __name__ == "__main__":
    train()
