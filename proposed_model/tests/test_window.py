from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proposed_model.data.loader import TrafficDataset
from data.preprocessing import DataPreprocessor
from data.scaler import FeatureScaler
from data.window import WindowGenerator

dataset = TrafficDataset()

df = dataset.load()

df = DataPreprocessor(df).process()

df = FeatureScaler(
    save_directory=Path("checkpoints")
).fit_transform(df)

generator = WindowGenerator(df)

X, Y = generator.generate()

print("Input Shape :", X.shape)
print("Target Shape:", Y.shape)

print()

print("First Input Window Shape :", X[0].shape)

print("First Target Shape :", Y[0].shape)