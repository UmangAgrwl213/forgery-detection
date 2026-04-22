# plot_losses.py

import os
import json
import matplotlib.pyplot as plt


LOG_FILE = "outputs/log.json"


def main():
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError("log.json not found. Run training first.")

    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    train = logs.get("train_loss", [])
    val = logs.get("val_loss", [])

    if len(train) == 0:
        raise RuntimeError("No data in log file.")

    plt.figure(figsize=(8, 5))

    plt.plot(train, label="Train Loss")
    if len(val) > 0:
        plt.plot(val, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()