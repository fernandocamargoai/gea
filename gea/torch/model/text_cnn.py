import torch
from future_shot.module import EmbeddingsDropout
from torch import nn

_METADATA_DIM = 39


class TextCNN(nn.Module):
    def __init__(
        self,
        sequence_embedding_dim: int,
        vocab_size: int = len("ATGCN"),
        num_filters: int = 64,
        depth: int = 1,
        kernel_sizes: list[int] = [3, 4, 5],
        dense_hidden_dim: int = 512,
        dropout_after_sequences: float = 0.0,
        dropout_after_convs: float = 0.0,
        dropout_after_sequence_fc: float = 0.0,
        dropout_before_output_fc: float = 0.0,
        dropout_for_embeddings: float = 0.0,
        factorization_dim: int = 64,
        activation: nn.Module = nn.SELU(),
    ):
        super().__init__()

        self.embeddings_dropout = EmbeddingsDropout(dropout_for_embeddings)
        self.activation = activation
        self.dropout_module = (
            nn.AlphaDropout if isinstance(activation, nn.SELU) else nn.Dropout
        )

        self.sequence_embedding = nn.Embedding(vocab_size, sequence_embedding_dim)

        self.dropout_after_sequences = nn.Dropout2d(dropout_after_sequences)

        self.convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            sequence_embedding_dim if i == 0 else num_filters,
                            num_filters,
                            k,
                        )
                        for i in range(depth)
                    ]
                )
                for k in kernel_sizes
            ]
        )
        convs_output_dim = len(kernel_sizes) * num_filters
        self.dropout_after_convs = self.dropout_module(dropout_after_convs)

        if dense_hidden_dim > 0:
            self.sequence_fc = nn.Linear(
                convs_output_dim,
                dense_hidden_dim,
            )
            self.dropout_after_sequence_fc = self.dropout_module(
                dropout_after_sequence_fc
            )
        self.dropout_before_output_fc = self.dropout_module(dropout_before_output_fc)

        self.sequence_output_fc = nn.Linear(
            (dense_hidden_dim if dense_hidden_dim > 0 else convs_output_dim)
            + _METADATA_DIM,
            factorization_dim,
        )

    def convs_and_max_pool(self, convs: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for conv in convs:
            x = self.activation(conv(x))
        return x.permute(0, 2, 1).max(1)[0]

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = self.sequence_embedding(batch["sequences"])

        x = self.dropout_after_sequences(x)
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.convs_and_max_pool(convs, x) for convs in self.convs], 1)
        x = self.dropout_after_convs(x)
        if hasattr(self, "sequence_fc"):
            x = self.activation(self.sequence_fc(x))
            x = self.dropout_after_sequence_fc(x)
        x = torch.cat((x, batch["metadata"]), 1)
        x = self.dropout_before_output_fc(x)
        x = self.sequence_output_fc(x)

        return x
