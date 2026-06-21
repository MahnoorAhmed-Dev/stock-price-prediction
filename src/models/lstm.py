import torch
import torch.nn as nn

class LSTMModel(nn.Module):
  """
    Stacked LSTM model for multi-step stock price forecasting.
    Architecture:
    Input → LSTM layers → Dropout → Fully Connected → Output
    Input shape:  (batch_size, seq_len, num_features)
    e.g. (32, 60, 16) — 32 samples, 60 days, 16 features
    Output shape: (batch_size, forecast_horizon)
    e.g. (32, 30) — 32 samples, 30 day forecast
  """
  
  def __init__(
    self,
    # number of features 16 in this case
    input_size: int,
    # number of hidden units in each LSTM layer
    hidden_size: int = 128,
    # number of stacked LSTM layers
    num_layers: int = 2,
    # dropout probability to prevent overfitting
    dropout: float = 0.2,
    # number of future days to predict
    forecast_horizon: int = 30,
    # if True, reads sequence both forward & backward
    bidirectional = False
    
    
  ):
    # calling the parent class constructor first for PyTorch models
    super(LSTMModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    
    # Number of directions 2 is bidirectional, 1 if not
    # Used to correctly size the output of LSTM -> Linear Layer
    self.num_directions= 2 if bidirectional else 1
    
    # LSTM Layer
    # LSTM processes the input seq step by step maintaining a hidden state
    # that captures tempral dependencies across time
    
    # batch_first = True means input shape is (batch, seq, feats) instead of pytorch's default (seq, batch, feats)
    # dropout applies bq LSTM layers (not after last one)
    # only active when num_layers>1
    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout if num_layers > 1 else 0.0,
      bidirectional=bidirectional
    )
    
    
    # Dropout layer
    # applied after LSTM output to prevent overfitting
    # randomly zeros out neurons during training
    self.dropout = nn.Dropout(dropout)
    
    # Fully connected output layer
    # projects the LSTM's last hiddens state to our forecast horizon
    # Input size = hidden_size * num_directions
    # (bidirectional LSTM concatenates forward + backward hidden states)
    self.fc = nn.Linear(
      hidden_size * self.num_directions,
      forecast_horizon
    )
  
  def forward(self, x):
    """
    Forward pass through the network.
    Args:
    x: input tensor of shape (batch_size, seq_len, input_size)

    Returns:
    predictions of shape (batch_size, forecast_horizon)
"""
    # initialise hidden state
    # h0 = initial hidden state, c0= inital cell state
    # both start as zeros at the beginning of each batch
    # shape: num_layers * num_directions, batch_size, hiden_size
    batch_size = x.size(0)
    h0 = torch.zeros(
      self.num_layers * self.num_directions,
      batch_size,
      self.hidden_size
    ).to(x.device)
    # .to(x.device) ensures h0 is on same device (CPU/GPU) as input
    
    c0 = torch.zeros(
      self.num_layers * self.num_directions,
      batch_size,
      self.hidden_size
    ).to(x.device)
    
    # LSTM forward pass
    # lstm_out: output at every timestep -> shape(batch, seq_len, hidden*directions)
    # (h_n, c_n): final hidden and cell states (won't be using these directly)
    lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
    
    # Taking last timestamp output
    # output at last timestamp only matters because it has seen the entire input seq
    # shape: (batch_size, hidden_size * num_directions)
    # : -> select all samples in the batch
    # -1-> select the last time step in each sequence
    # : -> select all hidden features from that time step
    last_out = lstm_out[:, -1, :]
    
    # Applying dropout
    out = self.dropout(last_out)
    
    # projecting to forecast horizon
    # Linear layer maps hidden representation → 30 future price predictions
    # Shape: (batch_size, forecast_horizon)
    out = self.fc(out)
    return out
  
if __name__ == "__main__":
  # dummy batch forward pass to test
  batch_size = 32
  seq_len = 60
  num_features = 16
  forecast_horizon = 30
  
  # dummy input mimicking one batch from the DataLoader
  dummy_input = torch.randn(batch_size, seq_len, num_features)
  
  # instantiate model with our config values
  model = LSTMModel(
    input_size=num_features,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    forecast_horizon=forecast_horizon,
    bidirectional=False
  )
  # run forward pass
  output = model(dummy_input)
  
  # 32, 60, 16
  print(f"Input shape:  {dummy_input.shape}")
  # 32, 30
  print(f"Output shape: {output.shape}")
  print(f"\nModel architecture:\n{model}")
  
  # count total trainable parameters
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"\nTotal trainable parameters: {total_params:,}")