import os
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import ta

def load_raw_data(ticker: str ="AAPL", data_dir:str="data/raw") -> pd.DataFrame:
  """loading th raw CSV file i downloaded in download.py"""
  path=os.path.join(data_dir, f"{ticker}_data.csv")
  
  # parse the date column as actual datetime so pandas can work with it properly
  # index_col="Date" tells pandas to use the Date column as the index of the DataFrame, which is common for time series data
  # parse_dates=True tells pandas to automatically convert the Date column from strings to datetime objects, which allows for easier date-based indexing and manipulation later on
  df = pd.read_csv(path, index_col="Date", parse_dates=True)
  print(f"Loaded {len(df)} rows from {path}")
  return df

def add_technical_indicators(df: pd.DataFrame)-> pd.DataFrame:
  """
  Adding the technical indicators as extra input features for the model.
  Raw OHLCV alone doesn't capture market momentum or volatility well,
  these indicators give the model richer signals to leanrn from by summarizing price action in different ways. 
  
  """
  # Trend Indicators:
  
  
  # using ta to calculate technical indicators
  # Simple Moving Average (SMA): average closing price over last N days
  # Smoothing out noise: if price> SMA it is an uptrend signal
  # window=20 means we are looking at the average price over the last 20 days, 
  # which is a common choice for short-term trend analysis
  df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
  # window=50 is a common choice for medium-term trend analysis, 
  # it captures the average price over the last 50 days, 
  # which can help identify longer-term trends compared to the 20-day SMA
  df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
  
  # Exponential Moving Average (EMA): gives more weight to recent prices,
  # so it reacts faster to price changes than SMA
  df["EMA_20"]= ta.trend.ema_indicator(df["Close"], window=20)
  # don't need EMA_50 since it is similar to SMA_50 and we want to avoid redundant features
  
  # Momentum Indicators:
  
  # RSI (Relative Strength Index): measures speed of price changes, range 0-100
  # RSI > 70 = overbought (price may drop), RSI < 30 = oversold (price may rise)
  df["RSI"]= ta.momentum.rsi(df["Close"], window=14)
  
  # MACD: diff bw 12-day and 26-day EMA
  # When MACD crosses above signal line = bullish, below = bearish
  df["MACD"]= ta.trend.macd(df["Close"])
  df["MACD_signal"]=ta.trend.macd_signal(df["Close"])
  
  # Volatility Indicators:
  # Bollinger Bands: bands around SMA based on standard deviation
  # Wide bands = high volatility, narrow bands = low volatility
  df["BB_high"] = ta.volatility.bollinger_hband(df["Close"])
  df["BB_low"] = ta.volatility.bollinger_lband(df["Close"])
  df["BB_width"] = df["BB_high"] - df["BB_low"]
  # ATR (Average True Range): measures market volatility over 14 days
  # High ATR = big price swings, low ATR = calm market
  df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
  
  # Volume Indicators:
  # OBV (On-Balance Volume): running total of volume based on price direction
  # Rising OBV = buyers are in control, falling OBV = sellers dominating
  df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
  # Drop rows with NaN values that appear because indicators need
  # a warm-up period (e.g. SMA_50 needs 50 rows before it has a value)
  df = df.dropna()
  print(f"Shape after adding indicators: {df.shape}")
  return df

def normalize_data(df: pd.DataFrame, save_dir: str="data/processed"):
  """
  Scale all features to [0, 1] range using MinMaxScaler.
  Neural networks train much better when inputs are in a small, consistent range.
  Saving the scaler so i can inverse-transform predictions back to real prices later.
  """
  os.makedirs(save_dir, exist_ok=True)
  scaler=MinMaxScaler(feature_range=(0,1))
  # fitting the scaler on all the features and transforming them
  scaled_array= scaler.fit_transform(df.values)
  # Converting back to DataFrame so i can keep column names and date index
  scaled_df= pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
  # Saving the scaler to disk - MUST use the same scaler later to
  # inverse-transform model predictions back into real dollar prices
  scaler_path = os.path.join(save_dir, "scaler.pkl")
  with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
  print(f"Scaler saved to {scaler_path}")
  
  # saving processed DataFrame as CSV
  processed_path = os.path.join(save_dir, "AAPL_processed.csv")
  scaled_df.to_csv(processed_path)
  print(f"Processed data saved to {processed_path}")
  return scaled_df, scaler

def run_preprocessing(ticker:str = "AAPL"):
  """ Full preprocessing pipeline: load -> add indicators -> normalize -> save"""
  #loading raw OHLCV data
  df= load_raw_data(ticker)
  #adding technical indicators
  df =add_technical_indicators(df)
  # normalize everything [0,1]
  scaled_df, scaler = normalize_data(df)
  
  print(f"\nFinal processed shaper: {scaled_df.shape}")
  print(f"Features: {scaled_df.columns.tolist()}")
  print(scaled_df.tail())
  
  return scaled_df, scaler

if __name__== "__main__":
  run_preprocessing()

  