from typing import Dict, Any, List

import torch
from future_shot.data import FutureShotPreprocessing
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

PADDING_VALUE = 1000


class SentencePiecePreprocessing(FutureShotPreprocessing):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)

    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        tokenized_sequences = [
            torch.tensor(tokenized_sequence, dtype=torch.long)
            for tokenized_sequence in self.sp_model.encode(batch["sequences"])
        ]
        return {
            "sequences": pad_sequence(
                tokenized_sequences, batch_first=True, padding_value=PADDING_VALUE
            ),
            "metadata": torch.tensor(batch["metadata"], dtype=torch.float),
        }
