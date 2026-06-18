import os
import yfinance as yf
import pandas as pd
from datetime import datetime

def download_stock_data(
  ticker:str = "AAPL",
  start_date:str ="2010-01-01",
  end_date:str = None,
  save_dir:str = "data/raw"
  ) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance and save it as CSV.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format. If None, defaults to today.
        save_dir: Directory to save the downloaded CSV file.
        
    Returns:
        DataFrame with OHLCV (Open, High, Low, Close, Volume) data.
    
    """
    # Default end date to today if not provided
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # yfinance downloads OHLCV data directly from Yahoo Finance
    # auto_adjust=True adjust prices for stock splits and dividends automatically
    # this ensures that the data is consistent and ready for analysis without needing manual adjustments
    # so i don't have to do it manually later
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    # if ticker is wrong or Yahoo has no data, bail early
    if df.empty:
      raise ValueError(f"No data found for ticker {ticker}. Check the ticker symbol.")
    
    # yfinance sometimes returns a MultiIndex column (ticker, field)
    #  e.g. (AAPL, Close) instead of just Close
    # this flattens it back to single-level column names
    if isinstance(df.columns, pd.MultiIndex):
      df.columns= df.columns.get_level_values(0)
    
    # Rename the index column to "Date" for clarity 
    df.index.name = "Date"
    
    # Keep only the 5 core columns: Open, High, Low, Close (price), Volume (trading activity)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    # drop any rows with missing values (e.g. due to market holidays)
    df = df.dropna()
    
    # create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # save as CSV so i don't re-load it every time i run this project
    save_path = os.path.join(save_dir, f"{ticker}_data.csv")
    df.to_csv(save_path)
    print(f"Saved {len(df)} rows to {save_path}")
    # return the DataFrame in case i want to use it immediately without re-loading from CSV
    return df
  
  # This block only runs when i execute this file directly
  # e.g. python src/data/download.py
  # it wont run when this file is imported by another module
  # if __name__ == "main" is a common Python idiom to allow code to be run as a script or imported as a module without executing the script code
  # in this case the script code is `download_stock_data()` which will download the stock data and save it as CSV
if __name__ =="__main__":
    df=download_stock_data()
    print(df.tail())
    # (rows, columns) = df.shape
    print(f"\nShape : {df.shape}")
    # should be 5 columns (Open, High, Low, Close, Volume) and ~3000 rows for 13 years of daily data
    print(f"\nColumns : {df.columns}")
    # full history of stock data from 2010-01-01 to today, so the date range should reflect that
    # df.index[0] is the first date in the index (the earliest date) and df.index[-1] is the last date in the index (the most recent date)
    print(f"Data range: {df.index[0]} -> {df.index[-1]}")