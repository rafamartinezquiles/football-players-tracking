"""
config.py

This module centralizes configuration constants for the Football AI project.

You can adjust:
- Paths to videos
- Model IDs
- Class IDs for ball / goalkeeper / player / referee
- HuggingFace model names, etc.

This keeps "magic numbers" and identifiers out of the logic code.
"""

import os
from pathlib import Path

# Base directory for the project
# You can customize this if your layout is different.
BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Video configuration
# ----------------------------------------------------------------------

# Default path to the main sample video used in the examples.
# You can change this to point to other videos.
DEFAULT_VIDEO_PATH = str(BASE_DIR / ".." / "0bfacc_0.mp4")
ALT_VIDEO_PATH = str(BASE_DIR / ".." / "121364_0.mp4")

# For convenience, choose one as the source for demos
SOURCE_VIDEO_PATH = ALT_VIDEO_PATH

# ----------------------------------------------------------------------
# Roboflow model configuration
# ----------------------------------------------------------------------

# Roboflow detection model for:
# - ball
# - goalkeeper
# - player
# - referee
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"

# Roboflow keypoint model for football field / pitch
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"

# ----------------------------------------------------------------------
# Class IDs used by the Roboflow detection model
# ----------------------------------------------------------------------

# IMPORTANT: These must match how the Roboflow model was trained.
# If your model uses different IDs, change them here and everything else
# should continue to work.
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# ----------------------------------------------------------------------
# Embeddings / clustering configuration
# ----------------------------------------------------------------------

# Name of the SigLIP visual model on HuggingFace
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

# Number of UMAP components for visualization
UMAP_N_COMPONENTS = 3

# Number of KMeans clusters for team splitting
KMEANS_NUM_CLUSTERS = 2

# Batch size for embedding extraction
EMBEDDING_BATCH_SIZE = 32

# ----------------------------------------------------------------------
# Ball tracking configuration
# ----------------------------------------------------------------------

# Number of previous homography matrices used to smooth ball projection
HOMOGRAPHY_SMOOTHING_WINDOW = 5

# Maximum allowed step size for ball tracking between frames.
# Used to filter out outliers in pitch coordinates.
MAX_BALL_DISTANCE_THRESHOLD = 500.0

# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------

# Default device for Torch-based models.
# This is just a string; the actual device is chosen in code that imports this.
DEFAULT_TORCH_DEVICE = os.environ.get("FOOTBALL_AI_DEVICE", "cuda")
