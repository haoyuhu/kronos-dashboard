from dataclasses import dataclass
from typing import Dict, Callable, Optional
import pandas as pd

from pathlib import Path

# Reuse existing loader implementations to avoid duplication
from core.data_loader import fetch_binance_data, fetch_akshare_data


@dataclass
class DataSource:
    name: str
    fetch_func: Callable[[str, str, int], Optional[pd.DataFrame]]


class DataSourceManager:
    def __init__(self) -> None:
        self.sources: Dict[str, DataSource] = {
            'binance': DataSource('binance', fetch_binance_data),
            'akshare': DataSource('akshare', fetch_akshare_data),
        }

    def available(self):
        return list(self.sources.keys())

    def has(self, name: str) -> bool:
        return name in self.sources

    def fetch(self, name: str, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        if name not in self.sources:
            raise KeyError(f"Unknown data source: {name}")
        return self.sources[name].fetch_func(symbol, interval, limit)