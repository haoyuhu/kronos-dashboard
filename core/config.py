from dataclasses import dataclass
from pathlib import Path
from configparser import ConfigParser


@dataclass
class CommonConfig:
    pred_horizon: int
    n_predictions: int
    hist_points: int
    vol_window: int
    update_interval_minutes: int


@dataclass
class ModelConfig:
    model_size: str  # one of: mini, small, base
    device: str      # cpu or cuda (if available)
    max_context: int # context length per model


class ConfigManager:
    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / 'config' / 'config.ini'

    def load(self) -> ConfigParser:
        parser = ConfigParser()
        parser.read(self.config_path)
        return parser

    def common(self) -> CommonConfig:
        parser = self.load()
        geti = parser.getint
        section = 'common'
        return CommonConfig(
            pred_horizon=geti(section, 'pred_horizon', fallback=24),
            n_predictions=geti(section, 'n_predictions', fallback=30),
            hist_points=geti(section, 'hist_points', fallback=200),
            vol_window=geti(section, 'vol_window', fallback=180),
            update_interval_minutes=geti(section, 'update_interval_minutes', fallback=60)
        )

    def model(self) -> ModelConfig:
        parser = self.load()
        section = 'model'
        gets = parser.get
        geti = parser.getint
        size = gets(section, 'model_size', fallback='small').strip().lower()
        # Default max context per model
        if size == 'mini':
            max_ctx = 2048
        else:
            max_ctx = 512
        device = gets(section, 'device', fallback='cpu').strip()
        return ModelConfig(model_size=size, device=device, max_context=max_ctx)