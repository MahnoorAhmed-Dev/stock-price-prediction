import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    load_scaler,
    load_trained_model,
    get_predictions,
    inverse_transform_predictions
)
from src.data.dataset import load_processed_data, split_data, create_dataloaders


def plot_loss_curves(results_dir: str = "outputs/results", plots_dir: str = "outputs/plots"):
    """
    Plot training and validation loss curves for both models.
    Helps visualize how well training went — are the models converging?
    Is there overfitting (train loss much lower than val loss)?
    """
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")

    for i, model_type in enumerate(["lstm", "transformer"]):
        # Load loss history saved during training
        history_path = os.path.join(results_dir, f"{model_type}_history.json")
        with open(history_path, "r") as f:
            history = json.load(f)

        train_losses = history["train_losses"]
        val_losses   = history["val_losses"]
        epochs       = range(1, len(train_losses) + 1)

        ax = axes[i]
        ax.plot(epochs, train_losses, label="Train Loss", color="steelblue",  linewidth=2)
        ax.plot(epochs, val_losses,   label="Val Loss",   color="orangered", linewidth=2, linestyle="--")

        # Mark the best val loss epoch
        best_epoch = np.argmin(val_losses) + 1
        best_loss  = min(val_losses)
        ax.axvline(x=best_epoch, color="green", linestyle=":", alpha=0.7, label=f"Best epoch ({best_epoch})")

        ax.set_title(f"{model_type.upper()} Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved to {save_path}")


def plot_predictions(
    model_type: str,
    preds_real: np.ndarray,
    targets_real: np.ndarray,
    plots_dir: str = "outputs/plots",
    num_samples: int = 5
):
    """
    Plot predicted vs actual prices for a few test samples.
    Each subplot shows one prediction window:
    - X axis: days into the forecast horizon (1-30)
    - Y axis: AAPL price in dollars
    - Blue line: actual price
    - Red dashed line: model prediction

    Args:
        num_samples: how many example windows to plot
    """
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    fig.suptitle(f"{model_type.upper()} — Predicted vs Actual Prices (Test Samples)", 
                 fontsize=13, fontweight="bold")

    # Pick evenly spaced samples from the test set
    indices = np.linspace(0, len(preds_real) - 1, num_samples, dtype=int)

    for plot_idx, sample_idx in enumerate(indices):
        ax = axes[plot_idx]

        actual = targets_real[sample_idx]    # true prices for this window
        predicted = preds_real[sample_idx]   # model predictions for this window
        days = range(1, len(actual) + 1)     # day 1 to day 30

        ax.plot(days, actual,    label="Actual",    color="steelblue", linewidth=2)
        ax.plot(days, predicted, label="Predicted", color="orangered", linewidth=2, linestyle="--")

        # Shade the area between actual and predicted to show error
        ax.fill_between(days, actual, predicted, alpha=0.15, color="gray")

        ax.set_title(f"Sample {sample_idx}", fontsize=10)
        ax.set_xlabel("Forecast Day")
        ax.set_ylabel("Price ($)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{model_type}_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction plot saved to {save_path}")


def plot_model_comparison(metrics_path: str = "outputs/results/metrics.json",
                          plots_dir: str = "outputs/plots"):
    """
    Bar chart comparing LSTM vs Transformer on MAE, RMSE, MAPE.
    Makes it easy to see which model performs better at a glance.
    """
    os.makedirs(plots_dir, exist_ok=True)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    models    = [m["model"].upper() for m in metrics]
    mae_vals  = [m["MAE"]  for m in metrics]
    rmse_vals = [m["RMSE"] for m in metrics]
    mape_vals = [m["MAPE"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("LSTM vs Transformer — Test Set Metrics", fontsize=14, fontweight="bold")

    colors = ["steelblue", "orangered"]

    # MAE bar chart
    axes[0].bar(models, mae_vals, color=colors, edgecolor="black", width=0.5)
    axes[0].set_title("MAE (Mean Absolute Error)")
    axes[0].set_ylabel("Dollars ($)")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mae_vals):
        axes[0].text(i, v + 0.3, f"${v:.2f}", ha="center", fontweight="bold")

    # RMSE bar chart
    axes[1].bar(models, rmse_vals, color=colors, edgecolor="black", width=0.5)
    axes[1].set_title("RMSE (Root Mean Squared Error)")
    axes[1].set_ylabel("Dollars ($)")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v + 0.3, f"${v:.2f}", ha="center", fontweight="bold")

    # MAPE bar chart
    axes[2].bar(models, mape_vals, color=colors, edgecolor="black", width=0.5)
    axes[2].set_title("MAPE (Mean Absolute % Error)")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mape_vals):
        axes[2].text(i, v + 0.1, f"{v:.2f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def run_visualization(config_path: str = "configs/config.yaml"):
    """Run all visualizations — loss curves, predictions, model comparison."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg   = config["data"]
    seq_cfg    = config["sequence"]
    train_cfg  = config["training"]
    output_cfg = config["outputs"]

    plots_dir   = output_cfg["plots_dir"]
    results_dir = output_cfg["results_dir"]
    model_dir   = output_cfg["model_dir"]

    # ── Plot 1: Loss curves ───────────────────────────────────────────────
    print("Generating loss curves...")
    plot_loss_curves(results_dir, plots_dir)

    # ── Load data for prediction plots ────────────────────────────────────
    data = load_processed_data(data_cfg["processed_dir"], data_cfg["ticker"])
    train_data, val_data, test_data = split_data(data, data_cfg["train_ratio"], data_cfg["val_ratio"])
    _, _, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_cfg["seq_len"],
        forecast_horizon=seq_cfg["forecast_horizon"],
        batch_size=train_cfg["batch_size"]
    )

    scaler = load_scaler("data/processed/scaler.pkl")

    # ── Plot 2 & 3: Prediction plots for each model ───────────────────────
    for model_type in ["lstm", "transformer"]:
        print(f"Generating prediction plots for {model_type.upper()}...")
        model  = load_trained_model(model_type, config, data.shape[1], model_dir)
        preds, targets = get_predictions(model, test_loader)
        preds_real, targets_real = inverse_transform_predictions(
            preds, targets, scaler, seq_cfg["target_col"]
        )
        plot_predictions(model_type, preds_real, targets_real, plots_dir)

    # ── Plot 4: Model comparison bar chart ────────────────────────────────
    print("Generating model comparison chart...")
    plot_model_comparison(
        metrics_path=os.path.join(results_dir, "metrics.json"),
        plots_dir=plots_dir
    )

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    run_visualization()