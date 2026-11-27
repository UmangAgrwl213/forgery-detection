# plot_losses.py

import argparse
import json
import os
from pathlib import Path

# Avoid a protected user-profile cache when training in a managed environment.
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))
import matplotlib

# Training can run without an interactive desktop/Tcl installation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_training_plots(log_file, output_file=None, show=False):
    """Create a persistent training report from the JSON written by main.py."""
    log_file = Path(log_file)
    if not log_file.exists():
        raise FileNotFoundError(f"Training log not found: {log_file}. Run training first.")
    with log_file.open("r", encoding="utf-8") as f:
        logs = json.load(f)

    train = logs.get("train_loss", [])
    val = logs.get("val_loss", [])

    if len(train) == 0:
        raise RuntimeError("No data in log file.")

    epochs = range(1, len(train) + 1)
    figure, (loss_ax, metric_ax) = plt.subplots(1, 2, figsize=(13, 5))
    loss_ax.plot(epochs, train, marker="o", label="Train loss")
    if val:
        loss_ax.plot(range(1, len(val) + 1), val, marker="o", label="Validation loss")
    loss_ax.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    if logs.get("val_positive_mask_f1"):
        metric_keys = {"val_positive_mask_f1": "Positive-mask F1", "val_positive_mask_iou": "Positive-mask IoU",
                       "val_empty_target_mean_predicted_area": "Highlighted area on empty masks"}
    else:
        metric_keys = {"val_iou": "IoU", "val_f1": "F1", "val_precision": "Precision", "val_recall": "Recall"}
    for key, label in metric_keys.items():
        values = logs.get(key, [])
        if values:
            metric_ax.plot(range(1, len(values) + 1), values, marker="o", label=label)
    metric_ax.set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Validation localization metrics")
    metric_ax.grid(alpha=0.3)
    if metric_ax.lines:
        metric_ax.legend()
    figure.tight_layout()

    output_file = Path(output_file or log_file.with_name("training_curves.png"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    print(f"Saved training curves: {output_file}")
    if show:
        plt.show()
    plt.close(figure)
    return output_file


def save_ai_training_plot(log_file, output_file=None):
    """Plot AI training loss and the trade-off at the validation cutoff."""
    log_file = Path(log_file)
    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No completed AI training epochs")
    epochs = [row["epoch"] for row in rows]
    figure, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(13, 5))
    loss_ax.plot(epochs, [row["train_loss"] for row in rows], marker="o")
    loss_ax.set(xlabel="Epoch", ylabel="Loss", title="AI detector training loss", xticks=epochs)
    for field, label in (("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced accuracy"),
                         ("generated_recall", "Generated recall"), ("authentic_false_positive_rate", "Authentic false alarms")):
        score_ax.plot(epochs, [row[field] for row in rows], marker="o", label=label)
    score_ax.set(xlabel="Epoch", ylabel="Score", title="AI detector validation", ylim=(0, 1), xticks=epochs)
    score_ax.legend()
    for axis in (loss_ax, score_ax):
        axis.grid(alpha=0.3)
    output = Path(output_file or log_file.with_name("training_curves.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training curves: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot saved forgery-model training curves.")
    parser.add_argument("--log", default="outputs/log.json", help="Path to training log JSON")
    parser.add_argument("--output", default=None, help="PNG output path (default: next to log)")
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window")
    args = parser.parse_args()
    if Path(args.log).suffix.lower() == ".jsonl":
        save_ai_training_plot(args.log, args.output)
    else:
        save_training_plots(args.log, args.output, args.show)


if __name__ == "__main__":
    main()
