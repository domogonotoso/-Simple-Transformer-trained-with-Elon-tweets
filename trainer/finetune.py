# finetune.py
# Knowledge Distillation for GPTMini from GPT2-small

"""
Finetune GPTMini via knowledge distillation from pretrained GPT2-small.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import yaml
import os

from transformers import GPT2LMHeadModel
from model.tokenizer import MyTokenizer
from model.transformer import GPTMini
from utils.plot_loss import plot_loss

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config['batch_size']
BLOCK_SIZE = config['block_size']
EPOCHS = config['epochs']
LR = float(config['learning_rate'])
VAL_SPLIT = config.get('val_split', 0.1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEMPERATURE = config.get('temperature', 2.0)  # softening factor


# Dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            tokens = tokenizer.encode(line.strip())
            if len(tokens) > BLOCK_SIZE:
                for i in range(0, len(tokens) - BLOCK_SIZE):
                    x = tokens[i:i + BLOCK_SIZE]
                    self.data.append(torch.tensor(x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Finetuning via distillation
def finetune():
    tokenizer = MyTokenizer()
    dataset = TextDataset('data/processed.txt', tokenizer)

    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty. Check block_size, tokenizer, or preprocessing.")

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Load teacher model (frozen)
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    teacher.eval()

    # Load student model
    student = GPTMini(vocab_size=tokenizer.vocab_size).to(DEVICE)
    optimizer = AdamW(student.parameters(), lr=LR)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0

        for step, x in enumerate(train_loader):
            x = x.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(x).logits
                teacher_soft = F.softmax(teacher_logits / TEMPERATURE, dim=-1)

            student_logits = student(x)
            student_log_soft = F.log_softmax(student_logits / TEMPERATURE, dim=-1)

            loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (TEMPERATURE ** 2)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate(student, val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), "checkpoints/best_student.pth")
            print("💾 Best student model saved.")

    torch.save(student.state_dict(), "checkpoints/last_student.pth")
    print("✅ Final student model saved.")

    plot_loss(train_losses, val_losses, save_path="results/distill_loss_plot.png")


# Validation loss (for student only)
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    finetune()
