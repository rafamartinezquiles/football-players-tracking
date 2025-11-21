"""
models.py

This module centralizes model initialization:

- Roboflow detection models (players, ball, goalkeeper, referee).
- Roboflow field keypoint model (pitch keypoints).
- HuggingFace SigLIP vision model and processor for embeddings.

By keeping model loading in one place, you can:
- Control lazy loading,
- Swap models easily,
- Avoid multiple copies of the same large model in memory.

All heavy imports are here so the rest of the code can stay lighter.
"""

import os
from typing import Tuple

import torch
from inference import get_model
from transformers import AutoProcessor, SiglipVisionModel

from config import (
    PLAYER_DETECTION_MODEL_ID,
    FIELD_DETECTION_MODEL_ID,
    SIGLIP_MODEL_NAME,
    DEFAULT_TORCH_DEVICE,
)
from project_secrets import load_api_keys


def init_roboflow_models():
    """
    Initialize Roboflow inference models.

    Returns:
        player_detection_model: object detection model for players/ball/etc.
        field_detection_model: keypoint detection model for pitch keypoints.
    """
    # Ensure API keys are loaded and set in environment
    _, roboflow_api_key = load_api_keys()
    os.environ["ROBOFLOW_API_KEY"] = roboflow_api_key

    # Build the player detection model
    player_detection_model = get_model(
        model_id=PLAYER_DETECTION_MODEL_ID,
        api_key=roboflow_api_key,
    )

    # Build the field (pitch) keypoint detection model
    field_detection_model = get_model(
        model_id=FIELD_DETECTION_MODEL_ID,
        api_key=roboflow_api_key,
    )

    return player_detection_model, field_detection_model


def init_siglip_embedding_model(
    device: str = DEFAULT_TORCH_DEVICE,
) -> Tuple[SiglipVisionModel, AutoProcessor]:
    """
    Initialize the SigLIP vision model and processor.

    This is used for extracting visual embeddings for each player crop.

    Args:
        device: Torch device string ("cuda", "cpu", "mps", etc.).

    Returns:
        model:      SiglipVisionModel instance on the specified device.
        processor:  AutoProcessor instance for preprocessing images.
    """
    hf_token, _ = load_api_keys()
    os.environ["HF_TOKEN"] = hf_token

    # Load the model and processor from HuggingFace
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, use_auth_token=hf_token)
    model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME, use_auth_token=hf_token)

    # Move model to the desired device (GPU if available and configured)
    model = model.to(device)

    return model, processor


def get_torch_device(preferred: str = DEFAULT_TORCH_DEVICE) -> torch.device:
    """
    Convert a simple string like "cuda" or "cpu" into a torch.device,
    handling the case where CUDA may not actually be available.

    Args:
        preferred: Desired device name ("cuda" or "cpu").

    Returns:
        torch.device instance.
    """
    if preferred == "cuda" and not torch.cuda.is_available():
        # Fallback to CPU if requested GPU is not available.
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(preferred)
