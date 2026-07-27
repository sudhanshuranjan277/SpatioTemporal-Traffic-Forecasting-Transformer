from pathlib import Path
import sys

# Add project root first
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import project modules
from proposed_model.data.loader import TrafficDataset
from data.preprocessing import DataPreprocessor
from data.scaler import FeatureScaler


dataset = TrafficDataset()

df = dataset.load()

df = DataPreprocessor(df).process()

scaler = FeatureScaler(
    save_directory=Path("checkpoints")
)

scaled = scaler.fit_transform(df)

scaler.save()

print("Original")
print(df.head())

print()

print("Scaled")
print(scaled.head())

print()

print("Scaler Saved Successfully")