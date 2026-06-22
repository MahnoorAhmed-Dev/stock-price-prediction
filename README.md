# Stock Price Prediction — LSTM vs Transformer

Forecasting **AAPL stock prices 30 days into the future** using deep learning models trained on Yahoo Finance historical data (2010–2026).

Built as a Deep Learning project comparing two architectures:
- **LSTM** — sequential model that captures temporal dependencies
- **Transformer** — attention-based model that learns long-range patterns

---

## Results

| Model | MAE | RMSE | MAPE | Best Val Loss | Epochs |
|-------|-----|------|------|--------------|--------|
| **LSTM** | $40.98 | $45.36 | 16.84% | 0.001567 | 16 |
| Transformer | $48.86 | $55.25 | 20.06% | 0.003139 | 27 |

> LSTM outperforms Transformer on this dataset — consistent with literature showing Transformers need larger datasets to shine.

---

## Project Structure

```
stock-price-prediction/
├── data/
│   ├── raw/                # downloaded AAPL OHLCV CSV
│   └── processed/          # normalized data + scaler
├── notebooks/
│   ├── EDA.ipynb           # exploratory data analysis
│   └── evaluation.ipynb    # model evaluation with plots
├── src/
│   ├── data/
│   │   ├── download.py     # Yahoo Finance data downloader
│   │   ├── preprocess.py   # technical indicators + normalization
│   │   └── dataset.py      # PyTorch Dataset + DataLoader
│   ├── models/
│   │   ├── lstm.py         # stacked LSTM architecture
│   │   └── transformer.py  # Transformer with positional encoding
│   ├── training/
│   │   ├── trainer.py      # train/val loop + early stopping
│   │   └── train.py        # training entry point
│   └── evaluation/
│       ├── metrics.py      # MAE, RMSE, MAPE + inverse transform
│       └── visualize.py    # loss curves, prediction plots
├── configs/
│   └── config.yaml         # all hyperparameters in one place
├── outputs/
│   ├── models/             # saved model weights (.pth)
│   ├── plots/              # generated charts
│   └── results/            # metrics JSON + loss history
├── requirements.txt
└── environment.yml
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Create conda environment
```bash
conda create -n stock-pred python=3.12
conda activate stock-pred
pip install -r requirements.txt
```

---

## Usage

### Download data
```bash
python src/data/download.py
```

### Preprocess
```bash
python src/data/preprocess.py
```

### Train
```bash
# Train LSTM
python -m src.training.train lstm

# Train Transformer
python -m src.training.train transformer
```

### Evaluate
```bash
python -m src.evaluation.metrics
```

### Visualize
```bash
python -m src.evaluation.visualize
```

---

## Models

### LSTM
- 2 stacked LSTM layers with hidden size 128
- Dropout (0.2) for regularization
- Fully connected output layer → 30 day forecast
- ~210,718 trainable parameters
- Early stopping triggered at epoch 16

### Transformer
- Linear input projection → d_model (64)
- Sinusoidal positional encoding
- 3 encoder layers with 4 attention heads
- Pre-norm architecture for stable training
- ~200K trainable parameters
- Early stopping triggered at epoch 27

---

## Features Used (16 total)

| Category | Features |
|----------|----------|
| Price | Open, High, Low, Close, Volume |
| Trend | SMA_20, SMA_50, EMA_20 |
| Momentum | RSI, MACD, MACD_signal |
| Volatility | BB_high, BB_low, BB_width, ATR |
| Volume | OBV |

---

## 🔧 Configuration

All hyperparameters live in `configs/config.yaml`:

```yaml
data:
  ticker: "AAPL"
  start_date: "2010-01-01"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

sequence:
  seq_len: 60           # days of history as input
  forecast_horizon: 30  # days to predict ahead
  target_col: 3         # Close price column index

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  patience: 10
  min_delta: 0.0001
  grad_clip: 1.0

lstm:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  bidirectional: false

transformer:
  d_model: 64
  nhead: 4
  num_encoder_layers: 3
  dim_feedforward: 256
  dropout: 0.1
  max_seq_len: 512
```

---

## Dataset Stats

| Stat | Value |
|------|-------|
| Ticker | AAPL |
| Date Range | 2010-01-04 → 2026-06-17 |
| Total Rows | 4,139 trading days |
| After Indicator Warmup | 4,090 rows |
| Train Set | 2,863 samples |
| Val Set | 613 samples |
| Test Set | 614 samples |
| Input Features | 16 |
| Lookback Window | 60 days |
| Forecast Horizon | 30 days |

---

## Key Findings

- **LSTM converged faster** (16 epochs vs 27) and achieved lower validation loss
- **Transformer needs more data** — its advantage shows at scale (millions of samples)
- **30-day forecasting is hard** — both models capture trends but struggle with sudden market movements
- **Technical indicators helped** — adding RSI, MACD, Bollinger Bands improved signal quality over raw OHLCV alone
- **Val loss comparison**: LSTM (0.001567) vs Transformer (0.003139) — LSTM 2x better

---

## Built With

- Python 3.12
- PyTorch 2.3.0
- yfinance 0.2.54
- scikit-learn 1.5.0
- ta 0.11.0
- pandas 2.2.2
- matplotlib 3.9.0
- seaborn 0.13.2
