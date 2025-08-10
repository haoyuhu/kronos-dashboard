import pandas as pd
from binance.client import Client
import akshare as ak
from typing import Optional

def _process_market_data(df: pd.DataFrame, timestamp_col: str, unit: Optional[str] = 'ms') -> pd.DataFrame:
    """Helper function to process raw market data from different sources."""
    df['timestamps'] = pd.to_datetime(df[timestamp_col], unit=unit)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    return df[['timestamps'] + numeric_cols]

def fetch_binance_data(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical k-line data from Binance.

    Args:
        symbol: The trading symbol (e.g., 'BTCUSDT').
        interval: The interval of k-lines (e.g., '1h', '4h', '1d').
        limit: The number of data points to retrieve.

    Returns:
        A pandas DataFrame with the market data, or None if an error occurs.
    """
    print(f"Fetching {limit} bars of {symbol} {interval} data from Binance...")
    try:
        client = Client()
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            print(f"No data returned from Binance for {symbol}.")
            return None

        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        df.rename(columns={'quote_asset_volume': 'amount'}, inplace=True)
        
        processed_df = _process_market_data(df, 'open_time')
        print("Binance data fetched successfully.")
        return processed_df
    except Exception as e:
        print(f"An error occurred while fetching Binance data for {symbol}: {e}")
        return None

def fetch_akshare_data(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical stock data from AkShare.

    Args:
        symbol: The stock symbol (e.g., '600519').
        interval: The data interval (unused for AkShare daily, but kept for consistency).
        limit: The number of data points to retrieve.

    Returns:
        A pandas DataFrame with the market data, or None if an error occurs.
    """
    print(f"Fetching {limit} bars of {symbol} data from AkShare...")
    try:
        # Note: AkShare's interval mapping might be needed here if more than daily is required.
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq").tail(limit)
        if stock_zh_a_hist_df.empty:
            print(f"No data returned from AkShare for {symbol}.")
            return None

        stock_zh_a_hist_df.rename(columns={'日期': 'timestamps_col', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount'}, inplace=True)
        
        processed_df = _process_market_data(stock_zh_a_hist_df, 'timestamps_col', unit=None)
        print("AkShare data fetched successfully.")
        return processed_df
    except Exception as e:
        print(f"An error occurred while fetching AkShare data for {symbol}: {e}")
        return None