import os
import sys
import yaml
import json
import torch
import numpy as np

# Add project root to path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_processed_data, split_data, create_dataloaders
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.training.trainer import Trainer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load hyperparameters from config.yaml.
    This way we never hardcode values in training code —
    everything is controlled from one central config file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Config loaded from {config_path}")
    return config


def build_model(model_type: str, config: dict, input_size: int) -> torch.nn.Module:
    """
    Instantiate either LSTM or Transformer based on model_type argument.
    All hyperparameters come from config.yaml.

    Args:
        model_type: "lstm" or "transformer"
        config:     full config dict loaded from yaml
        input_size: number of input features (16 in our case)

    Returns:
        instantiated model ready for training
    """
    seq_cfg   = config["sequence"]
    
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
        print(f"Built LSTM model")

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
        print(f"Built Transformer model")

    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose 'lstm' or 'transformer'.")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    return model


def run_training(model_type: str = "lstm", config_path: str = "configs/config.yaml"):
    """
    Full training pipeline:
    1. Load config
    2. Load & split data
    3. Create DataLoaders
    4. Build model
    5. Train
    6. Save results
    """

    # ── Step 1: Load config ───────────────────────────────────────────────
    config = load_config(config_path)

    # Extract config sections for convenience
    data_cfg     = config["data"]
    seq_cfg      = config["sequence"]
    train_cfg    = config["training"]
    output_cfg   = config["outputs"]

    # ── Step 2: Load & split data ─────────────────────────────────────────
    data = load_processed_data(
        data_dir=data_cfg["processed_dir"],
        ticker=data_cfg["ticker"]
    )

    train_data, val_data, test_data = split_data(
        data,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"]
    )

    # ── Step 3: Create DataLoaders ────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_cfg["seq_len"],
        forecast_horizon=seq_cfg["forecast_horizon"],
        batch_size=train_cfg["batch_size"]
    )

    # ── Step 4: Build model ───────────────────────────────────────────────
    # input_size = number of features in our processed data
    input_size = data.shape[1]  # 16 features
    model = build_model(model_type, config, input_size)

    # ── Step 5: Train ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=train_cfg["learning_rate"],
        epochs=train_cfg["epochs"],
        patience=train_cfg["patience"],
        min_delta=train_cfg["min_delta"],
        grad_clip=train_cfg["grad_clip"],
        model_dir=output_cfg["model_dir"],
        model_name=model_type  # saves as lstm_best.pth or transformer_best.pth
    )

    history = trainer.train()

    # ── Step 6: Save results ──────────────────────────────────────────────
    # Save loss history as JSON so we can plot it later in evaluation
    os.makedirs(output_cfg["results_dir"], exist_ok=True)
    results_path = os.path.join(output_cfg["results_dir"], f"{model_type}_history.json")

    with open(results_path, "w") as f:
        json.dump({
            "model": model_type,
            "train_losses": history["train_losses"],
            "val_losses":   history["val_losses"],
            "best_val_loss": history["best_val_loss"]
        }, f, indent=2)

    print(f"\nLoss history saved to {results_path}")
    return history


if __name__ == "__main__":
    # Default to LSTM — pass "transformer" as argument to train Transformer
    # Usage:
    #   python src/training/train.py           → trains LSTM
    #   python src/training/train.py transformer → trains Transformer
    model_type = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    print(f"\n{'='*60}")
    print(f"  Training: {model_type.upper()}")
    print(f"{'='*60}")
    run_training(model_type=model_type)