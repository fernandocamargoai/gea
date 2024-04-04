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

        self.preprocessing_fn: SentencePiecePreprocessing = (
            self.model_ref.custom_objects["preprocessing_fn"]
        )
        self.class_index_mapping: Dict[str, int] = self.model_ref.custom_objects[
            "class_index_mapping"
        ]
        self.index_class_mapping: Dict[int, str] = self.model_ref.custom_objects[
            "index_class_mapping"
        ]

    def _preprocess(self, inputs: list[Input]) -> dict[str, torch.Tensor]:
        sequences = []
        metadata = []
        for input_ in inputs:
            sequences.append(input_.sequence)
            metadata.append(input_.metadata.vector())
        batch = {"sequences": sequences, "metadata": metadata}
        batch = self.preprocessing_fn(batch)
        batch["sequences"] = batch["sequences"].to(self.device_id)
        batch["metadata"] = batch["metadata"].to(self.device_id)

        return batch

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=1000)
    async def predict_scores(self, inputs: list[Input]) -> np.ndarray:
        batch = self._preprocess(inputs)

        with torch.inference_mode():
            return self.model(batch).cpu().detach().numpy()

    @bentoml.api(
        batchable=True, batch_dim=(0, 0), max_batch_size=32, max_latency_ms=1000
    )
    async def encode(self, inputs: list[Input]) -> np.ndarray:
        batch = self._preprocess(inputs)

        with torch.inference_mode():
            return self.model.compute_embeddings(batch).cpu().detach().numpy()

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
        new_index = (
            self.class_index_mapping[lab_id]
            if lab_id in self.class_index_mapping
            else len(self.index_class_mapping)
        )
        self.model.insert_class(
            new_index, torch.tensor(embedding, device=self.device_id)
        )
        self.index_class_mapping[new_index] = lab_id
        self.class_index_mapping[lab_id] = new_index


@bentoml.service
class GeneticEngineeringAttributionService(object):
    future_shot_service: FutureShotService = bentoml.depends(FutureShotService)

    @bentoml.api
    async def predict(self, sequence: str, metadata: Metadata, k: int = 10) -> Output:
        scores = await self.future_shot_service.predict_scores(
            [Input(sequence=sequence, metadata=metadata)]
        )

        return await self.future_shot_service.generate_output(scores, k)

    @bentoml.api
    async def new_lab(self, inputs: list[Input], lab_id: str):
        embeddings = await self.future_shot_service.encode(inputs)
        lab_embedding = np.mean(embeddings, axis=0)
        await self.future_shot_service.insert_lab(lab_embedding, lab_id)
