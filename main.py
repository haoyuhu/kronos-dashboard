#!/usr/bin/env python3
"""
Kronos Financial Forecasting System

This module provides a comprehensive financial forecasting system using the Kronos foundation model.
It supports multiple data sources, generates probabilistic forecasts, and creates professional 
visualizations for web deployment.

Usage:
    python main.py [options]

Examples:
    # Run with mock data
    python main.py --mock

    # Run with real data and model inference  
    python main.py

    # Run continuously with scheduler
    python main.py --schedule

Author: Kronos Team
License: MIT
"""

import argparse
import schedule
import time
import sys
from pathlib import Path
from typing import Dict, Optional

from core.config import ConfigManager
from core.data_source import DataSourceManager
from core.model_manager import ModelManager
from core.engine import ForecastingEngine
from core.web_generator import WebGenerator
from utils.logger import logger


class KronosForecastingApp:
    """
    Main application class for the Kronos forecasting system.
    
    Orchestrates data fetching, model inference, chart generation, and web deployment
    in a clean, maintainable architecture.
    """
    
    def __init__(self, repo_path: Path, mock: bool = False):
        """Initialize the forecasting application."""
        self.repo_path = Path(repo_path)
        self.config_manager = ConfigManager(self.repo_path)
        self.data_manager = DataSourceManager()
        # Load configuration early
        self.config = self.config_manager.load()
        self.common_config = self.config_manager.common()
        self.model_config = self.config_manager.model()

        # Initialize components
        self.model_manager = ModelManager(
            device=self.model_config.device,
            max_context=self.model_config.max_context,
            model_size=self.model_config.model_size,
        )
        self.forecasting_engine = ForecastingEngine(self.repo_path)
        self.web_generator = WebGenerator(self.repo_path)
        
        # CLI-controlled mock mode
        self.mock = mock
        
        # Initialize model if not in mock mode
        self.model_loaded = False
        
    def _ensure_model_loaded(self) -> None:
        """Lazy load the model when needed."""
        if not self.model_loaded and not self.mock:
            logger.info(f"Loading Kronos model (size={self.model_config.model_size}, device={self.model_config.device})...")
            self.model_manager.load()  # Uses default HF cache
            self.model_loaded = True
            logger.info("Model loaded successfully")
    
    def _run_mock_mode(self) -> Dict[str, dict]:
        """Run in mock mode using existing charts (PNG only)."""
        logger.info("Running in mock mode...")
        charts_dir = self.repo_path / 'docs' / 'static' / 'chart'
        
        if not charts_dir.exists():
            logger.warning(f"Charts directory not found: {charts_dir}")
            return {}
        
        metrics = {}
        # PNG placeholders only
        chart_files = list(charts_dir.glob('*.png'))
        for chart_file in chart_files:
            symbol = chart_file.stem
            metrics[symbol] = {
                "upside_prob": 0.5,  # Mock data
                "vol_amp_prob": 0.5,  # Mock data
                "chart_path": str((self.repo_path / 'static' / 'chart' / f"{symbol}.png").relative_to(self.repo_path))
            }
        
        if not metrics:
            logger.warning("No existing charts found for mock mode")
        else:
            logger.info(f"Found {len(metrics)} symbols in mock mode")
        
        return metrics
    
    def _run_inference_mode(self) -> Dict[str, dict]:
        """Run with real model inference."""
        self._ensure_model_loaded()
        predictor = self.model_manager.predictor()
        metrics = {}
        
        # Process each configured data source
        for source_name in self.data_manager.available():
            if source_name not in self.config:
                continue
                
            source_config = self.config[source_name]
            symbols = [s.strip() for s in source_config['symbols'].split(',')]
            interval = source_config['interval']
            
            logger.info(f"Processing {len(symbols)} symbols from {source_name}")
            
            for symbol in symbols:
                logger.info(f"Forecasting {symbol} from {source_name}")
                
                # Fetch data
                total_points = self.common_config.hist_points + self.common_config.vol_window
                df_full = self.data_manager.fetch(source_name, symbol, interval, total_points)
                
                if df_full is None or df_full.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Run forecasting
                result = self.forecasting_engine.run_for_symbol(
                    df_full=df_full,
                    predictor=predictor,
                    symbol=symbol,
                    pred_horizon=self.common_config.pred_horizon,
                    n_predictions=self.common_config.n_predictions,
                    hist_points=self.common_config.hist_points,
                    vol_window=self.common_config.vol_window,
                    interval=interval
                )
                
                if result:
                    metrics[symbol] = {
                        "upside_prob": result.upside_prob,
                        "vol_amp_prob": result.vol_amp_prob,
                        "chart_path": str(result.chart_path.relative_to(self.repo_path / 'docs'))
                    }
                    logger.info(f"Completed forecast for {symbol}")
                else:
                    logger.warning(f"Failed to generate forecast for {symbol}")
        
        return metrics
    
    def run_forecast(self) -> bool:
        """Execute a single forecasting cycle."""
        logger.info("=" * 60)
        logger.info(f"Starting forecast at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        try:
            # Generate forecasts
            if self.mock:
                metrics = self._run_mock_mode()
            else:
                metrics = self._run_inference_mode()
            
            if not metrics:
                logger.warning("No forecast data generated, skipping web update")
                return False
            
            # Update web page
            output_path = self.web_generator.update(metrics)
            logger.info(f"Updated web page: {output_path}")
            logger.info(f"Generated forecasts for {len(metrics)} symbols")
            
            logger.info("=" * 60)
            logger.info("Forecast cycle completed successfully")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Forecast cycle failed: {e}")
            return False
    
    def run_scheduler(self) -> None:
        """Run the forecasting system on a schedule."""
        interval = self.common_config.update_interval_minutes
        logger.info(f"Scheduling forecasts every {interval} minutes")
        
        schedule.every(interval).minutes.do(self.run_forecast)
        
        while True:
            schedule.run_pending()
            time.sleep(1)


def main():
    """Main entry point for the forecasting application."""
    parser = argparse.ArgumentParser(
        description="Kronos Financial Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mock             # Run once in mock mode
  %(prog)s                    # Run once with real model inference
  %(prog)s --schedule         # Run continuously on schedule
        """
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='Run in mock mode using existing PNG charts at web/static/chart'
    )
    parser.add_argument(
        '--schedule', 
        action='store_true',
        help='Run continuously on schedule'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize application
        repo_path = Path(__file__).resolve().parent
        app = KronosForecastingApp(repo_path, mock=args.mock)
        
        # Run application
        if args.schedule:
            app.run_scheduler()
        else:
            success = app.run_forecast()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()