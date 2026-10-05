import pandas as pd
import numpy as np
import torch

print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
