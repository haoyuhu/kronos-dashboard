from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

from core.pipeline import make_prediction, calculate_metrics


@dataclass
class ForecastResult:
    upside_prob: float
    vol_amp_prob: float
    chart_path: Path


class ForecastingEngine:
    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path)

    def run_for_symbol(
        self,
        df_full: pd.DataFrame,
        predictor,
        symbol: str,
        pred_horizon: int,
        n_predictions: int,
        hist_points: int,
        vol_window: int,
        interval: str,
    ) -> Optional[ForecastResult]:
        if df_full is None or df_full.empty:
            print(f"No data for {symbol}")
            return None

        df_for_model = df_full.iloc[:-1]
        close_preds, volume_preds, v_close_preds = make_prediction(
            df_for_model, predictor, pred_horizon, n_predictions, interval
        )
        hist_df_for_plot = df_for_model.tail(hist_points)
        hist_df_for_metrics = df_for_model.tail(vol_window)

        upside_prob, vol_amp_prob = calculate_metrics(
            hist_df_for_metrics, close_preds, v_close_preds, vol_window
        )
        from core.pipeline import create_plot

        chart_path = create_plot(
            hist_df_for_plot,
            close_preds,
            volume_preds,
            symbol,
            pred_horizon,
            interval,
            self.repo_path,
        )
        return ForecastResult(upside_prob, vol_amp_prob, chart_path)
