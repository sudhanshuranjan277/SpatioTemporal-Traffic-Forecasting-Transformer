from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.embedding import TrafficEmbedding
from configs.config import (
    NUM_INPUT_FEATURES,
    EMBEDDING_DIM,
)


def test_embedding():

    batch = 8
    history = 12
    nodes = 9

    sample = torch.randn(
        batch,
        history,
        nodes,
        NUM_INPUT_FEATURES,
    )

    model = TrafficEmbedding(
        max_sequence_length=history
    )

    output = model(sample)

    print("=" * 60)
    print("Input Shape :", sample.shape)
    print("Output Shape:", output.shape)
    print("=" * 60)

    assert output.shape == (
        batch,
        history,
        nodes,
        EMBEDDING_DIM,
    )

    print("✓ Embedding Test Passed")


if __name__ == "__main__":
    test_embedding()