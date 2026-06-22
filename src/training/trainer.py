import os
import time
import torch
import torch.nn as nn
import numpy as np


class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Why early stopping?
    - Prevents overfitting — model memorizing training data instead of learning patterns
    - Saves time — no point training further if val loss is getting worse
    - Automatically saves the best model checkpoint before stopping
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        """
        Args:
            patience:  how many epochs to wait for improvement before stopping
            min_delta: minimum change to count as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0               # counts epochs without improvement
        self.best_loss = float("inf")  # tracks the best val loss seen so far
        self.should_stop = False       # flag — True when training should stop

    def __call__(self, val_loss: float) -> bool:
        """
        Call after each epoch with the current validation loss.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found — reset counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement — increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Handles the full training loop for both LSTM and Transformer models.

    Responsibilities:
    - Run train and validation loops each epoch
    - Track train/val losses over time
    - Save the best model checkpoint
    - Early stopping when val loss plateaus
    - Print training progress
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        learning_rate: float = 0.001,
        epochs: int = 50,
        patience: int = 10,
        min_delta: float = 0.0001,
        grad_clip: float = 1.0,
        model_dir: str = "outputs/models",
        model_name: str = "model"
    ):
        """
        Args:
            model:         the LSTM or Transformer model to train
            train_loader:  DataLoader for training data
            val_loader:    DataLoader for validation data
            learning_rate: step size for Adam optimizer
            epochs:        maximum number of training epochs
            patience:      early stopping patience
            min_delta:     minimum improvement for early stopping
            grad_clip:     max gradient norm (prevents exploding gradients)
            model_dir:     where to save model checkpoints
            model_name:    filename prefix for saved model
        """

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")

        # Move model to device (GPU or CPU)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.model_dir = model_dir
        self.model_name = model_name

        # ── Loss Function ─────────────────────────────────────────────────
        # MSE (Mean Squared Error) — standard for regression tasks
        # Penalizes large errors more than small ones (squares the difference)
        self.criterion = nn.MSELoss()

        # ── Optimizer ─────────────────────────────────────────────────────
        # Adam: adaptive learning rate optimizer — works well out of the box
        # for most deep learning tasks without much tuning
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # ── Learning Rate Scheduler ───────────────────────────────────────
        # ReduceLROnPlateau: reduces learning rate when val loss plateaus
        # factor=0.5 → halves the LR | patience=5 → waits 5 epochs
        # This helps the model fine-tune as it gets closer to convergence
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # ── Early Stopping ────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        # ── History ───────────────────────────────────────────────────────
        # Track losses for plotting later
        self.train_losses = []
        self.val_losses = []

        # Create model output directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def _train_epoch(self) -> float:
        """
        Run one full pass through the training data.
        Returns average training loss for this epoch.
        """

        # Set model to training mode — enables dropout and batch norm
        self.model.train()
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            # Move batch to same device as model
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients from previous batch
            # (PyTorch accumulates gradients by default)
            self.optimizer.zero_grad()

            # Forward pass — get predictions
            predictions = self.model(inputs)

            # Compute loss between predictions and true targets
            loss = self.criterion(predictions, targets)

            # Backward pass — compute gradients
            loss.backward()

            # Gradient clipping — cap gradient norm to prevent exploding gradients
            # especially important for LSTMs which can have unstable gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Update model weights using computed gradients
            self.optimizer.step()

            total_loss += loss.item()

        # Return average loss per batch
        return total_loss / len(self.train_loader)

    def _val_epoch(self) -> float:
        """
        Run one full pass through the validation data.
        Returns average validation loss for this epoch.
        """

        # Set model to evaluation mode — disables dropout
        # so we get deterministic predictions
        self.model.eval()
        total_loss = 0.0

        # torch.no_grad() disables gradient computation — saves memory and speeds up val
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self) -> dict:
        """
        Full training loop — runs for self.epochs epochs or until early stopping.
        Returns a dict with train/val loss history and best val loss.
        """

        print(f"\nStarting training for up to {self.epochs} epochs...")
        print("-" * 60)

        best_val_loss = float("inf")
        best_model_path = os.path.join(self.model_dir, f"{self.model_name}_best.pth")

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            # Run train and validation passes
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()

            # Store losses for plotting later
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Step the LR scheduler based on val loss
            self.scheduler.step(val_loss)

            # Save model if this is the best val loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                saved_marker = " ✓ saved"
            else:
                saved_marker = ""

            # Print epoch summary
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:3d}/{self.epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
                f"{saved_marker}"
            )

            # Check early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                print(f"No improvement for {self.early_stopping.patience} consecutive epochs.")
                break

        print("-" * 60)
        print(f"Training complete. Best val loss: {best_val_loss:.6f}")
        print(f"Best model saved to: {best_model_path}")

        return {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "best_val_loss": best_val_loss,
            "best_model_path": best_model_path
        }