from typing import Dict, Any, List

import torch
from future_shot.data import FutureShotPreprocessing
import sentencepiece as spm


class SentencePiecePreprocessing(FutureShotPreprocessing):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)

    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        return {
            "sequences": torch.tensor(
                self.sp_model.encode(batch["sequences"]), dtype=torch.long
            ),
            "metadata": torch.tensor(batch["metadata"], dtype=torch.float),
        }
