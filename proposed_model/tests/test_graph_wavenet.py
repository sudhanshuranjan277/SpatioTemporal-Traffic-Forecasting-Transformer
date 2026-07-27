from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs.config import EMBEDDING_DIM
from models.graph_wavenet import GraphWaveNet


def test_graph_wavenet():

    batch_size = 4
    history = 12
    nodes = 9

    x = torch.randn(
        batch_size,
        history,
        nodes,
        EMBEDDING_DIM,
    )

    adjacency = torch.rand(
        batch_size,
        nodes,
        nodes,
    )

    adjacency = torch.softmax(
        adjacency,
        dim=-1,
    )

    model = GraphWaveNet()

    output = model(
        x,
        adjacency,
    )

    assert output.shape == (
        batch_size,
        history,
        nodes,
        EMBEDDING_DIM,
    )

    print("✓ Output Shape :", output.shape)


def test_invalid_input():

    model = GraphWaveNet()

    try:

        x = torch.randn(
            4,
            12,
            EMBEDDING_DIM,
        )

        adjacency = torch.rand(
            4,
            9,
            9,
        )

        model(
            x,
            adjacency,
        )

    except ValueError:

        print("✓ Input validation passed.")

    else:

        raise AssertionError(
            "Input validation failed."
        )


if __name__ == "__main__":

    print("=" * 60)
    print("Running GraphWaveNet Tests")
    print("=" * 60)

    test_graph_wavenet()
    test_invalid_input()

    print("=" * 60)
    print("All Tests Passed Successfully!")
    print("=" * 60)