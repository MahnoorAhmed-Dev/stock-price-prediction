import os
import sys
import json
import pickle
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_processed_data, split_data, create_dataloaders
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel


def load_scaler(scaler_path: str = "data/processed/scaler.pkl"):
    """
    Load the MinMaxScaler we saved during preprocessing.
    We need this to convert normalized predictions (0-1)
    back into real dollar prices.
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")
    return scaler


def load_trained_model(model_type: str, config: dict, input_size: int, model_dir: str):
    """
    Rebuild model architecture and load saved weights from training.

    Args:
        model_type: "lstm" or "transformer"
        config:     full config dict
        input_size: number of input features (16)
        model_dir:  directory where .pth files are saved

    Returns:
        model with trained weights loaded, set to eval mode
    """
    seq_cfg = config["sequence"]

    if model_type == "lstm":
        lstm_cfg = config["lstm"]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            forecast_horizon=seq_cfg["forecast_horizon"],
            bidirectional=lstm_cfg["bidirectional"]
        )

    elif model_type == "transformer":
        tf_cfg = config["transformer"]
        model = TransformerModel(
            input_size=input_size,
            d_model=tf_cfg["d_model"],
            nhead=tf_cfg["nhead"],
            num_encoder_layers=tf_cfg["num_encoder_layers"],
            dim_feedforward=tf_cfg["dim_feedforward"],
            dropout=tf_cfg["dropout"],
            forecast_horizon=seq_cfg["forecast_horizon"],
            max_seq_len=tf_cfg["max_seq_len"]
        )

    # Load saved weights into model architecture
    model_path = os.path.join(model_dir, f"{model_type}_best.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Set to eval mode — disables dropout for deterministic predictions
    model.eval()
    print(f"Loaded {model_type} weights from {model_path}")
    return model


def get_predictions(model, test_loader, device="cpu"):
    """
    Run model on test set and collect all predictions and true targets.

    Returns:
        preds:   numpy array of model predictions, shape (num_samples, forecast_horizon)
        targets: numpy array of true values,       shape (num_samples, forecast_horizon)
    """
    all_preds   = []
    all_targets = []

    # No gradient computation needed during inference
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)

            # Forward pass — get predictions for this batch
            predictions = model(inputs)

            # Move to CPU and convert to numpy for metric calculation
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

    # Stack all batches into single arrays
    preds   = np.concatenate(all_preds,   axis=0)  # (num_samples, forecast_horizon)
    targets = np.concatenate(all_targets, axis=0)  # (num_samples, forecast_horizon)

    return preds, targets


def inverse_transform_predictions(preds, targets, scaler, target_col: int = 3):
    """
    Convert normalized predictions back to real dollar prices.

    The scaler was fit on all 16 features together.
    To inverse transform just the Close price column, we need to:
    1. Create a dummy array with all 16 features (zeros for non-target cols)
    2. Put our predictions in the Close column
    3. Inverse transform the whole array
    4. Extract just the Close column

    Args:
        preds:      normalized predictions, shape (num_samples, forecast_horizon)
        targets:    normalized true values, shape (num_samples, forecast_horizon)
        scaler:     fitted MinMaxScaler from preprocessing
        target_col: index of Close price column (3)

    Returns:
        preds_real, targets_real in original dollar price scale
    """
    num_features = scaler.n_features_in_  # 16

    def inverse_col(arr):
        """Inverse transform a single column (forecast_horizon values per sample)."""
        results = []
        for row in arr:
            # Create dummy array of zeros with all 16 features
            dummy = np.zeros((len(row), num_features))
            # Put our values in the Close price column
            dummy[:, target_col] = row
            # Inverse transform — scaler converts all 16 cols back
            inversed = scaler.inverse_transform(dummy)
            # Extract only the Close price column
            results.append(inversed[:, target_col])
        return np.array(results)

    preds_real   = inverse_col(preds)
    targets_real = inverse_col(targets)

    return preds_real, targets_real


def calculate_metrics(preds_real, targets_real, model_type: str):
    """
    Calculate evaluation metrics on real dollar prices.

    Metrics:
    - MAE  (Mean Absolute Error):       average absolute difference in dollars
    - RMSE (Root Mean Squared Error):   like MAE but penalizes large errors more
    - MAPE (Mean Absolute % Error):     percentage error — scale independent

    Args:
        preds_real:   predictions in real dollar prices
        targets_real: true values in real dollar prices
        model_type:   "lstm" or "transformer" (for printing)

    Returns:
        dict with MAE, RMSE, MAPE
    """
    # Flatten from (num_samples, forecast_horizon) to 1D for metric calculation
    preds_flat   = preds_real.flatten()
    targets_flat = targets_real.flatten()

    # MAE — average absolute error in dollars
    mae = np.mean(np.abs(preds_flat - targets_flat))

    # RMSE — square root of mean squared error
    # more sensitive to large prediction errors than MAE
    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))

    # MAPE — mean absolute percentage error
    # avoid division by zero with a small epsilon
    mape = np.mean(np.abs((targets_flat - preds_flat) / (targets_flat + 1e-8))) * 100

    print(f"\n{'='*40}")
    print(f"  {model_type.upper()} — Test Set Metrics")
    print(f"{'='*40}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"{'='*40}\n")

    return {"model": model_type, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_evaluation(config_path: str = "configs/config.yaml"):
    """
    Full evaluation pipeline for both models:
    1. Load data & create test loader
    2. Load trained models
    3. Get predictions
    4. Inverse transform to real prices
    5. Calculate & save metrics
    """
    import yaml

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg   = config["data"]
    seq_cfg    = config["sequence"]
    train_cfg  = config["training"]
    output_cfg = config["outputs"]

    # Load data and create test loader
    data = load_processed_data(data_cfg["processed_dir"], data_cfg["ticker"])
    train_data, val_data, test_data = split_data(data, data_cfg["train_ratio"], data_cfg["val_ratio"])
    _, _, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_cfg["seq_len"],
        forecast_horizon=seq_cfg["forecast_horizon"],
        batch_size=train_cfg["batch_size"]
    )

    # Load scaler for inverse transform
    scaler = load_scaler("data/processed/scaler.pkl")

    # Evaluate both models
    all_metrics = []
    for model_type in ["lstm", "transformer"]:
        print(f"\nEvaluating {model_type.upper()}...")

        # Load trained model
        model = load_trained_model(model_type, config, data.shape[1], output_cfg["model_dir"])

        # Get predictions on test set
        preds, targets = get_predictions(model, test_loader)

        # Convert from normalized → real dollar prices
        preds_real, targets_real = inverse_transform_predictions(
            preds, targets, scaler, seq_cfg["target_col"]
        )

        # Calculate metrics
        metrics = calculate_metrics(preds_real, targets_real, model_type)
        all_metrics.append(metrics)

    # Save metrics to JSON
    os.makedirs(output_cfg["results_dir"], exist_ok=True)
    metrics_path = os.path.join(output_cfg["results_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Print comparison summary
    print(f"\n{'='*40}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*40}")
    for m in all_metrics:
        print(f"  {m['model'].upper():12s} MAE: ${m['MAE']:.4f} | RMSE: ${m['RMSE']:.4f} | MAPE: {m['MAPE']:.2f}%")
    print(f"{'='*40}")

    return all_metrics


if __name__ == "__main__":
    run_evaluation()