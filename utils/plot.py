import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_loss(train_losses, val_losses=None, save_path='results/plot_loss.png', x_label='Epoch', show=False):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='orange')
    
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
