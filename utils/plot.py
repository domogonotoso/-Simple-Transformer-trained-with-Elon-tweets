import matplotlib.pyplot as plt
import os

def plot_loss(train_losses, save_path='results/plot_loss.png', x_label='Epoch', show=False):
    """
    Plot training loss curve and save it.

    Args:
        train_losses (list of float): Training loss per epoch or step.
        save_path (str): Path to save the plot.
        x_label (str): Label for the x-axis (default: 'Epoch').
        show (bool): If True, displays the plot interactively.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
