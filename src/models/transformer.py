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
      torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
      
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