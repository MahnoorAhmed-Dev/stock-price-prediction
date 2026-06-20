import numpy as np
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
  """ 
  PyTorch Dataset for stock price prediction.

  Converts our processed time series into (input_sequence, target_sequence) pairs
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