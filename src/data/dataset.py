import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class StockDataset(Dataset):
  """ 
  PyTorch Dataset for stock price prediction.

  Converts processed time series into (input_sequence, target_sequence) pairs
  using a sliding window approach:

  Example with seq_len=5, forecast_horizon=3:

  Index:  0  1  2  3  4  5  6  7
  Data:  [a, b, c, d, e, f, g, h]

  Window 1: input=[a,b,c,d,e]  target=[f,g,h]
  Window 2: input=[b,c,d,e,f]  target=[g,h,i]
  ... and so on

  This turns one flat time series into many training examples.
  """
  def __init__(
    self,
    data: np.ndarray,
    # how many past days the model looks at
    seq_len: int = 60,
    # how many future days to predict
    forecast_horizon :int = 30,
    # column index of 'Close' price
    target_col: int = 3
    
  ):
    self.data = data
    self.seq_len= seq_len
    self.forecast_horizon= forecast_horizon,
    self.target_col = target_col
    
    # Total sliding windows we can create from the data
    self.num_samples = len(data) - seq_len - forecast_horizon + 1
    
  def __len__(self):
    # PyTorch calls this to know how many samples exist
    return self.num_samples
  
  def __getitem__(self,idx):
    """
    Returns one (input, target) pair for a given index.

    input_seq:  all 16 features for seq_len days            → (seq_len, num_features)
    target_seq: only Close price for forecast_horizon days  → (forecast_horizon,)
    
    """
    # All features for the input window
    input_seq = self.data[idx: idx + self.seq_len]
    # Only close price column for the target window
    # Select the target values (labels) that the model should predict.
    target_seq = self.data[
      # Start right after the input sequence ends.
      # Example: if idx = 0 and seq_len = 24, start at index 24.
      idx + self.seq_len :

      # Stop after collecting 'forecast_horizon' number of future values.
      # Example: if forecast_horizon = 12, stop at index 36 (exclusive).
      idx + self.seq_len + self.forecast_horizon,

      # Select only the target column (e.g., temperature, sales, stock price)
      # instead of all feature columns.
      self.target_col
    ]
    # Convert to float 32 tensors for nns
    return (
      torch.tensor(input_seq, dtype=torch.float32),
      torch.tensor(target_seq, dtype=torch.float32)
      )
    
  def load_processed_data(data_dir: str = "data/processed", ticker: str= "AAPL") -> np.ndarray:
    """ 
    Load the processed CSV created in preprocess.py.
    Returns a raw numpy array — shape: (num_days, num_features)
    
    """
    path = os.path.join(data_dir, f"{ticker}_processed.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates = True)
    print(f"Loaded processed data: {df.shape}")
    
    # Convert to numpy - Dataset class works w arrays not DataFrames
    return df.values
  
  def split_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
    # test ratio is implicitly 1 - 0.7 - 0.15 = 0.15
    
  ) -> tuple:
    """
    Split time series into train / val / test sets.

    IMPORTANT: splitting by time, NOT randomly.
    Random splits cause data leakage — the model would see future data during training, 
    making results unrealistically optimistic.

    70 percent train, 15 percent val and 15 percent test
    """
    
    # Get the total number of rows (samples/time steps) in the dataset.
    n = len(data)

    # Calculate the index where the training dataset should end.
    # Example: If n = 1000 and train_ratio = 0.7,
    # then train_end = 700.
    train_end = int(n * train_ratio)

    # Calculate the index where the validation dataset should end.
    # This is after both the training and validation portions.
    # Example: If train_ratio = 0.7 and val_ratio = 0.15,
    # then val_end = 850.
    val_end = int(n * (train_ratio + val_ratio))

    # Select the training data from the beginning of the dataset
    # up to (but not including) train_end.
    train_data = data[:train_end]

    # Select the validation data starting where the training data ends
    # and ending just before val_end.
    val_data = data[train_end:val_end]

    # Select the remaining data as the test dataset.
    # This contains all samples after the validation set.
    test_data = data[val_end:]

    # Print the number of samples in each dataset
    # to verify that the split was performed correctly.
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Return the three datasets so they can be used
    # for model training, validation, and testing.
    return train_data, val_data, test_data
    