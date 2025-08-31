import torch

# --- Paths for Training Data ---
# These are used by train.py to find your sample dataset.
DATA_DIR = "data"
LABEL_FILE = "labels.txt"

# --- Model & Application Configuration ---
# This is the name of the file where the trained model weights will be saved.
MODEL_WEIGHTS = "model_weights.pth"

# Number of sentiment classes (e.g., 0: Negative, 1: Neutral, 2: Positive)
NUM_CLASSES = 3

# Training batch size. Kept small for your sample dataset.
BATCH_SIZE = 2

# Set the device for computation (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")