from train import train
from plotting import plot_metrics


if __name__ == "__main__":
    results = train()
    plot_metrics(metrics=results, save_path='training_plot.png')