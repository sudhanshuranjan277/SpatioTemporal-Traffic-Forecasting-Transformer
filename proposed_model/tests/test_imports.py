from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from configs.config import *
from proposed_model.data.loader import TrafficDataset
from data.preprocessing import DataPreprocessor
from models.embedding import TrafficEmbedding
from models.tsformer import TSFormer

print("✅ All imports successful.")
    
    