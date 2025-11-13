from typing import Any, Dict

from torch import nn


class MultiMediaXCLIPModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any], k: int = 10) -> None:
        super().__init__()
        self.model_config = model_config
        self.k = k

    def initialize(self, *args, **kwargs) -> None:
        pass

    def encode(self, *args, **kwargs) -> Any:
        pass

    def fit(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Any:
        pass

    def predict_most_similar(self, *args, **kwargs) -> Any:
        pass

    def forward(self):
        pass
