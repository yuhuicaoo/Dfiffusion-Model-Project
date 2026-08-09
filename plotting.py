import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics: dict[str, list], save_path=None):
    """
    Create metric plot from training results (losses, epoch time, learning rates)
    """

    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    learning_rates = metrics['learning_rates']
    epoch_times = metrics['epoch_times']

    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    ax1, ax2, ax3, ax4 = axes
    
    # Training & Val Loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(epochs, train_losses, label="Train Loss", linestyle="-", color='blue')
    ax1.plot(epochs, val_losses, label="Validation Loss", linestyle="-", color='orange')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Loss vs Epochs")

    # Log Loss
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Log(Loss)")
    ax2.plot(epochs, np.log(train_losses), label="Train Loss", linestyle="-", color='blue')
    ax2.plot(epochs, np.log(val_losses), label="Validation Loss", linestyle="-", color='orange')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Log(Loss) vs Epochs")

    # Epoch time
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time s")
    ax3.plot(epochs, epoch_times, label="Epoch Time", linestyle="-")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Time per Epoch")

    # Learning rate
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.plot(epochs, learning_rates, label="Learning Rate", linestyle="-")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Learning Rate vs Epochs")
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()