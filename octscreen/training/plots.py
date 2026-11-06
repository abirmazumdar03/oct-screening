# octscreen/training/plots.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(csv_path, out_png):
    h = pd.read_csv(csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_ce"], label="train CE")
    ax[0].plot(h["epoch"], h["val_loss"], label="val CE")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["epoch"], h["val_acc"], label="val accuracy")
    ax[1].plot(h["epoch"], h["val_macro_f1"], label="val macro-F1")
    ax[1].plot(h["epoch"], h["val_urgent_sens"], label="val urgent sensitivity")
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
