from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from model import KronosTokenizer, Kronos, KronosPredictor


@dataclass
class ModelArtifacts:
    tokenizer: KronosTokenizer
    model: Kronos
    predictor: KronosPredictor


class ModelManager:
    def __init__(
        self, device: str = "cpu", max_context: int = 512, model_size: str = "small"
    ) -> None:
        self.device = device
        self.max_context = max_context
        self.model_size = model_size
        self.artifacts: Optional[ModelArtifacts] = None

    def _hf_ids(self):
        size = (self.model_size or "small").lower()
        if size == "mini":
            tokenizer_id = "NeoQuasar/Kronos-Tokenizer-2k"
            model_id = "NeoQuasar/Kronos-mini"
        elif size == "base":
            tokenizer_id = "NeoQuasar/Kronos-Tokenizer-base"
            model_id = "NeoQuasar/Kronos-base"
        else:
            tokenizer_id = "NeoQuasar/Kronos-Tokenizer-base"
            model_id = "NeoQuasar/Kronos-small"
        return tokenizer_id, model_id

    def load(self, cache_dir: Optional[Path] = None) -> ModelArtifacts:
        # Use default HF cache when cache_dir is None
        print(
            f"Loading Kronos model (size={self.model_size}) from Hugging Face (using default cache)..."
        )
        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        tokenizer_id, model_id = self._hf_ids()
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id, **kwargs)
        model = Kronos.from_pretrained(model_id, **kwargs)
        tokenizer.eval()
        model.eval()
        predictor = KronosPredictor(
            model, tokenizer, device=self.device, max_context=self.max_context
        )
        self.artifacts = ModelArtifacts(tokenizer, model, predictor)
        print("Model loaded.")
        return self.artifacts

    def predictor(self) -> KronosPredictor:
        if not self.artifacts:
            raise RuntimeError("Model is not loaded. Call load() first.")
        return self.artifacts.predictor
