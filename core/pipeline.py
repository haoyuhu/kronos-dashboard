import time
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_prediction(df, predictor, pred_horizon, n_predictions, interval):
    """Generates probabilistic forecasts using the Kronos model."""
    last_timestamp = df['timestamps'].max()

    if 'd' in interval.lower():
        freq = 'D'
        td = pd.Timedelta(days=1)
    elif 'h' in interval.lower():
        freq = 'h'
        td = pd.Timedelta(hours=1)
    else:
        freq = 'min'
        td = pd.Timedelta(minutes=1)

    start_new_range = last_timestamp + td
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=pred_horizon,
        freq=freq
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    print("Making main prediction (T=0.6)...")
    begin_time = time.time()
    close_preds_main, volume_preds_main = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=pred_horizon, T=0.6, top_p=0.9,
        sample_count=n_predictions, verbose=True
    )
    print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")

    print("Making volatility prediction (T=0.9)...")
    begin_time = time.time()
    close_preds_volatility, _ = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=pred_horizon, T=0.9, top_p=0.9,
        sample_count=n_predictions, verbose=True
    )
    print(f"Volatility prediction completed in {time.time() - begin_time:.2f} seconds.")

    return close_preds_main, volume_preds_main, close_preds_volatility


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df, vol_window):
    """
    Calculates upside and volatility amplification probabilities for the 24h horizon.
    """
    last_close = hist_df['close'].iloc[-1]

    # 1. Upside Probability (for the 24-hour horizon)
    # This is the probability that the price at the end of the horizon is higher than now.
    final_hour_preds = close_preds_df.iloc[-1]
    upside_prob = (final_hour_preds > last_close).mean()

    # 2. Volatility Amplification Probability (over the 24-hour horizon)
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-vol_window:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    print(f"Upside Probability (24h): {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


from pathlib import Path

def create_plot(hist_df: pd.DataFrame, 
              close_preds_df: pd.DataFrame, 
              volume_preds_df: pd.DataFrame, 
              symbol: str, 
              pred_horizon: int, 
              interval: str, 
              repo_path: Path) -> Path:
    """Generates and saves a comprehensive forecast chart.

    Args:
        hist_df: DataFrame with historical data.
        close_preds_df: DataFrame with predicted close prices.
        volume_preds_df: DataFrame with predicted volumes.
        symbol: The symbol of the asset.
        pred_horizon: The prediction horizon.
        interval: The interval of the data.
        repo_path: The path to the repository.

    Returns:
        The path to the generated chart.
    """
    print(f"Generating comprehensive forecast chart for {symbol}...")
    try:
        # plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        if 'd' in interval.lower():
            time_unit = 'days'
            title_horizon_unit = 'Days'
        elif 'h' in interval.lower():
            time_unit = 'hours'
            title_horizon_unit = 'Hours'
        else: # default to minutes
            time_unit = 'minutes'
            title_horizon_unit = 'Minutes'

        hist_time = hist_df['timestamps']
        last_hist_time = hist_time.iloc[-1]
        
        pred_time = pd.to_datetime([last_hist_time + timedelta(**{time_unit: i + 1}) for i in range(len(close_preds_df))])

        ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price', linewidth=1.5)
        mean_preds = close_preds_df.mean(axis=1)
        ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
        ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1), color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
        ax1.set_title(f'{symbol} Probabilistic Price & Volume Forecast (Next {pred_horizon} {title_horizon_unit})', fontsize=16, weight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2.bar(hist_time, hist_df['volume'], color='skyblue', label='Historical Volume', width=0.03)
        ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown', label='Mean Forecasted Volume', width=0.03)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time (UTC)')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        separator_time = hist_time.iloc[-1] + timedelta(minutes=30)
        for ax in [ax1, ax2]:
            ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
            ax.tick_params(axis='x', rotation=30)

        fig.tight_layout()
        chart_dir = repo_path / 'docs' / 'static' / 'chart'
        chart_dir.mkdir(exist_ok=True)
        chart_path = chart_dir / f'{symbol}.png'
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        print(f"Chart saved to: {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error creating plot for {symbol}: {e}")
        return None