import pickle

import bentoml
import torch
from future_shot.model import FutureShotModel

from gea.preprocessing import SentencePiecePreprocessing
from gea.torch.model.text_cnn import TextCNN

if __name__ == "__main__":
    model = FutureShotModel(
        encoder=TextCNN(
            sequence_embedding_dim=200,
            vocab_size=1001,
            num_filters=256,
            depth=1,
            kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dense_hidden_dim=0,
            dropout_after_sequences=0.2,
            dropout_after_convs=0.0,
            dropout_after_sequence_fc=0.0,
            dropout_before_output_fc=0.0,
            dropout_for_embeddings=0.0,
            factorization_dim=200,
        ),
        embedding_dim=200,
        num_classes=1262 + 1,
        normalize_embeddings=True,
    )
    model.load_state_dict(torch.load("artifacts/model.pt"))
    model.eval()
    with open("artifacts/class_index_mapping.pkl", "rb") as f:
        class_index_mapping = pickle.load(f)

    preprocessing_fn = SentencePiecePreprocessing(model_path="artifacts/m1.model")

    bentoml.pytorch.save_model(
        "gea_model",
        model,
        signatures={"__call__": {"batchable": True}},
        labels={"owner": "fernandocamargo", "project": "gea"},
        custom_objects={
            "preprocessing_fn": preprocessing_fn,
            "index_class_mapping": {v: k for k, v in class_index_mapping.items()},
        },
    )
