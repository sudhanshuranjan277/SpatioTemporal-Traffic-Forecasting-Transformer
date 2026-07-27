from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs.config import EMBEDDING_DIM
from models.structure_generator import DynamicGraphGenerator


def test_structure_generator():

    batch_size = 4
    history = 12
    nodes = 9

    x = torch.randn(
        batch_size,
        history,
        nodes,
        EMBEDDING_DIM,
    )

    model = DynamicGraphGenerator()

    adjacency = model(x)

    assert adjacency.shape == (
        batch_size,
        nodes,
        nodes,
    )

    print("✓ Adjacency Shape :", adjacency.shape)


def test_row_sum():

    batch_size = 2
    history = 12
    nodes = 9

    x = torch.randn(
        batch_size,
        history,
        nodes,
        EMBEDDING_DIM,
    )

    model = DynamicGraphGenerator()

    adjacency = model(x)

    row_sum = adjacency.sum(dim=-1)

    print("Row Sum:")
    print(row_sum)


if __name__ == "__main__":

    print("=" * 60)
    print("Running Structure Generator Tests")
    print("=" * 60)

    test_structure_generator()
    test_row_sum()

    print("=" * 60)
    print("All Tests Passed")
    print("=" * 60)