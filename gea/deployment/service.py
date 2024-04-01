import logging
from typing import Dict, cast

import bentoml
import numpy as np

import torch
from future_shot.model import FutureShotModel

from gea.preprocessing import SentencePiecePreprocessing
from gea.deployment.schema import Output, ScoredLab, Metadata, Input

MODEL_NAME = "gea_model:latest"

logger = logging.getLogger(__name__)


@bentoml.service
class FutureShotService(object):
    model_ref = bentoml.models.get(MODEL_NAME)

    def __init__(self):
        if torch.cuda.is_available():
            self.device_id = "cuda"
            # by default, torch.FloatTensor will be used on CPU.
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            self.device_id = "cpu"
        self.model: FutureShotModel = cast(
            FutureShotModel,
            bentoml.pytorch.load_model(self.model_ref, device_id=self.device_id),
        )
        # We want to turn off dropout and batchnorm when running inference.
        self.model.train(False)

        self.index_class_mapping: Dict[str, int] = self.model_ref.custom_objects[
            "index_class_mapping"
        ]

    @bentoml.api(
        batchable=True, batch_dim=(0, 0), max_batch_size=32, max_latency_ms=1000
    )
    async def predict_scores(
        self, input_: tuple[torch.Tensor, torch.Tensor]
    ) -> np.ndarray:
        sequences, metadata = input_
        sequences = sequences.to(self.device_id)
        metadata = metadata.to(self.device_id)
        with torch.inference_mode():
            return (
                self.model({"sequences": sequences, "metadata": metadata})
                .cpu()
                .detach()
                .numpy()
            )

    @bentoml.api(
        batchable=True, batch_dim=(0, 0), max_batch_size=32, max_latency_ms=1000
    )
    async def encode(self, input_: tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        sequences, metadata = input_
        sequences = sequences.to(self.device_id)
        metadata = metadata.to(self.device_id)
        with torch.inference_mode():
            return (
                self.model.compute_embeddings(
                    {"sequences": sequences, "metadata": metadata}
                )
                .cpu()
                .detach()
                .numpy()
            )

    @bentoml.api
    async def generate_output(self, scores: np.ndarray, k: int) -> Output:
        top_k_idx = np.argsort(scores[0])[-k:][::-1]
        top_k_classes = [self.index_class_mapping[idx] for idx in top_k_idx]

        return Output(
            ranked_labs=[
                ScoredLab(id=class_, score=score)
                for class_, score in zip(top_k_classes, scores[0, top_k_idx].tolist())
            ]
        )

    @bentoml.api
    async def insert_lab(self, embedding: np.ndarray, lab_id: str):
        new_index = len(self.index_class_mapping)
        self.model.insert_class(
            new_index, torch.tensor(embedding, device=self.device_id)
        )
        self.index_class_mapping[new_index] = lab_id


@bentoml.service
class GeneticEngineeringAttributionService(object):
    future_shot_service: FutureShotService = bentoml.depends(FutureShotService)
    model_ref = bentoml.models.get(MODEL_NAME)

    def __init__(self):
        self.preprocessing_fn: SentencePiecePreprocessing = (
            self.model_ref.custom_objects["preprocessing_fn"]
        )

    @bentoml.api
    async def predict(self, sequence: str, metadata: Metadata, k: int = 10) -> Output:
        batch = {"sequences": [sequence], "metadata": [metadata.vector()]}
        batch = self.preprocessing_fn(batch)
        scores = await self.future_shot_service.predict_scores(
            (batch["sequences"], batch["metadata"])
        )

        return await self.future_shot_service.generate_output(scores, k)

    @bentoml.api
    async def new_lab(self, inputs: list[Input], lab_id: str):
        batch = {
            "sequences": [input_.sequence for input_ in inputs],
            "metadata": [input_.metadata.vector() for input_ in inputs],
        }
        batch = self.preprocessing_fn(batch)

        embeddings = await self.future_shot_service.encode(
            (batch["sequences"], batch["metadata"])
        )
        lab_embedding = np.mean(embeddings, axis=0)
        await self.future_shot_service.insert_lab(lab_embedding, lab_id)
