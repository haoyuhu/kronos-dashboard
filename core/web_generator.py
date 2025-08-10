from typing import Dict
from pathlib import Path
from datetime import datetime, timezone
from jinja2 import Environment, FileSystemLoader


class WebGenerator:
    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path)
        self.template_dir = self.repo_path / 'templates'
        self.web_dir = self.repo_path / 'docs'
        self.output_path = self.web_dir / 'index.html'

    def update(self, metrics: Dict[str, dict]) -> Path:
        # Ensure web directory exists
        self.web_dir.mkdir(exist_ok=True)
        
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template('index.jinja')

        now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        # Use provided chart paths as relative to web/
        forecast_data = {}
        for symbol, data in metrics.items():
            forecast_data[symbol] = {
                "upside_prob": f'{data["upside_prob"]:.1%}',
                "vol_amp_prob": f'{data["vol_amp_prob"]:.1%}',
                "chart_path": data["chart_path"],
            }

        rendered_html = template.render(
            update_time=now_utc_str,
            forecast_data=forecast_data
        )

        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        return self.output_path