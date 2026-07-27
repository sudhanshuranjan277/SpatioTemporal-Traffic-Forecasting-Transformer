from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs.config import EMBEDDING_DIM
from models.tsformer import TSFormer


def test_tsformer():

    batch = 8
    history = 12
    nodes = 9

    sample = torch.randn(
        batch,
        history,
        nodes,
        EMBEDDING_DIM,
    )

    model = TSFormer()

    output = model(sample)

    print("=" * 60)
    print("Input Shape :", sample.shape)
    print("Output Shape:", output.shape)
    print("=" * 60)

    assert output.shape == sample.shape

    print("✓ TSFormer Test Passed")


if __name__ == "__main__":
    test_tsformer()