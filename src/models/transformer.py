import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  """
    Injects position information into the input embeddings.

    Transformers have NO built-in sense of order — unlike LSTMs which process
    tokens one by one, transformers process all tokens simultaneously.
    Without positional encoding, [day1, day2, day3] looks the same as [day3, day1, day2].

    Using sine/cosine functions at different frequencies to encode position:
    - Even dimensions: sin(position / 10000^(2i/d_model))
    - Odd dimensions:  cos(position / 10000^(2i/d_model))

    This gives each position a unique fingerprint the model can learn from.
    """
  def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(dropout)
    
    # Matrix of shape (max_seq_len, d_model) to hold position encodings
    pe = torch.zeros(max_seq_len, d_model)
    
    # position indices: 0, 1 , 2, ...., max_seq_len -1 -> shape(max_seq_len, 1)
    position= torch.arange(0, max_seq_len).unsqueeze(1).float()
    
    # Scaling factor - creates different frequencies for each dimension pair
    # lower dimensions = low frequency (captures global/ slow patterns)
    # Higher Dimensions = high frequency (captures local/fast patterns)
    div_term = torch.exp(
      torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
      
    )
    # Even indices -> sine wave, Odd indices -> cosine waves
    pe[:, 0::2]= torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Adding batch dimensions -> (1, max_seq_len, d_model)
    # so it can broadcast across any batch size automatically
    pe= pe.unsqueeze(0)
    
    # Register as buffer : not a learnable param
    # but still moves to GPU automatically when model.to(device) is called
    self.register_buffer("pe", pe)
  
  def forward(self,x):
    """
    Add positional encoding to input embeddings.
    Args:
    x: input tensor of shape (batch_size, seq_len, d_model)
    Returns:
    x with positional information added, same shape
    """
    # Slice pe to match the actual seq length (may be shorter than max_seq_len)
    x = x + self.pe[:, : x.size(1), :]
    return self.dropout(x)
class TransformerModel(nn.Module):
    """
    Transformer-based model for multi-step stock price forecasting.

    Architecture:
        Input → Linear Projection → Positional Encoding →
        Transformer Encoder → Last Timestep → Fully Connected → Output

    Why Transformer for time series?
    - Self-attention captures long-range dependencies better than LSTM
    - Processes all timesteps in parallel (faster than sequential LSTM)
    - Multiple attention heads learn different temporal patterns simultaneously

    Input shape:  (batch_size, seq_len, num_features)  e.g. (32, 60, 16)
    Output shape: (batch_size, forecast_horizon)         e.g. (32, 30)
    """

    def __init__(
        self,
        input_size: int,               # number of input features (16)
        d_model: int = 64,             # internal embedding dimension
        nhead: int = 4,                # number of attention heads (d_model must be divisible by nhead)
        num_encoder_layers: int = 3,   # number of stacked transformer encoder blocks
        dim_feedforward: int = 256,    # size of feedforward network inside each encoder block
        dropout: float = 0.1,          # dropout rate
        forecast_horizon: int = 30,    # number of future days to predict
        max_seq_len: int = 512         # maximum sequence length for positional encoding
    ):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # ── Input Projection ──────────────────────────────────────────────
        # Transformers require all inputs to have dimension d_model
        # Our input has 16 features — project it up to d_model (64)
        # Think of this as an embedding layer for continuous inputs
        self.input_projection = nn.Linear(input_size, d_model)

        # ── Positional Encoding ───────────────────────────────────────────
        # Adds position-aware signal so model knows the order of timesteps
        # Uses the PositionalEncoding class we defined above
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # ── Transformer Encoder Layer ─────────────────────────────────────
        # One encoder layer = Multi-Head Self Attention + Feedforward Network
        # batch_first=True: input shape is (batch, seq, features)
        # norm_first=True:  pre-norm architecture — more stable training
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        # ── Stack Encoder Layers ──────────────────────────────────────────
        # num_encoder_layers blocks stacked on top of each other
        # each block refines the representation from the previous one
        # final LayerNorm stabilizes the output
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # ── Output Head ───────────────────────────────────────────────────
        # After encoder: shape is (batch, seq_len, d_model)
        # We take the last timestep → (batch, d_model)
        # then project to forecast_horizon → (batch, 30)
        self.fc = nn.Linear(d_model, forecast_horizon)

        # ── Weight Initialization ─────────────────────────────────────────
        # Xavier uniform initialization helps gradients flow better
        # at the start of training — avoids vanishing/exploding gradients
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through the Transformer.

        Args:
            x: input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            predictions of shape (batch_size, forecast_horizon)
        """

        # ── Project input features to d_model ────────────────────────────
        # (batch, seq_len, 16) → (batch, seq_len, 64)
        x = self.input_projection(x)

        # ── Scale embeddings ──────────────────────────────────────────────
        # Standard transformer trick: multiply by sqrt(d_model)
        # keeps values in a stable range before adding positional encoding
        x = x * math.sqrt(self.d_model)

        # ── Add positional encoding ───────────────────────────────────────
        # Each timestep now carries both its feature info and position info
        x = self.positional_encoding(x)

        # ── Pass through Transformer Encoder ─────────────────────────────
        # Self-attention: every timestep attends to every other timestep
        # learns which past days are most relevant for prediction
        # Output shape: (batch, seq_len, d_model)
        x = self.transformer_encoder(x)

        # ── Take last timestep output ─────────────────────────────────────
        # The last timestep has attended to the full sequence
        # so it carries a summary of everything the model learned
        # Shape: (batch, d_model)
        x = x[:, -1, :]

        # ── Project to forecast horizon ───────────────────────────────────
        # Shape: (batch, forecast_horizon) e.g. (32, 30)
        out = self.fc(x)

        return out
if __name__ == "__main__":
    # Sanity check — run a forward pass with dummy data
    # to verify shapes are correct before training

    batch_size = 32
    seq_len = 60
    num_features = 16
    forecast_horizon = 30

    # Dummy batch mimicking one batch from our DataLoader
    dummy_input = torch.randn(batch_size, seq_len, num_features)

    # Instantiate model with same values as config.yaml
    model = TransformerModel(
        input_size=num_features,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        forecast_horizon=forecast_horizon
    )

    # Run forward pass
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")  # expect (32, 60, 16)
    print(f"Output shape: {output.shape}")        # expect (32, 30)
    print(f"\nModel architecture:\n{model}")

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Compare with LSTM parameter count
    print(f"\nFor reference — LSTM had: 210,718 parameters")
    print(f"Transformer has:          {total_params:,} parameters")    