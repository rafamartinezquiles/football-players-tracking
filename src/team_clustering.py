"""
team_clustering.py

Responsible for grouping players into teams based on jersey colors
(or more generally, visual similarity of player crops).

Two approaches are demonstrated:

1) Manual pipeline:
   - Extract crops.
   - Compute SigLIP embeddings.
   - Run UMAP + KMeans.
   - (Optionally) visualize in 3D using Plotly.

2) Simplified pipeline using `sports.common.team.TeamClassifier`:
   - Automatically handles SigLIP + UMAP + KMeans internally.

Also includes:
- Heuristic to assign goalkeepers to teams based on spatial proximity.
"""

from typing import List, Tuple

import numpy as np
import supervision as sv
from tqdm import tqdm

from config import (
    PLAYER_ID,
    EMBEDDING_BATCH_SIZE,
    UMAP_N_COMPONENTS,
    KMEANS_NUM_CLUSTERS,
)
from models import get_torch_device, init_siglip_embedding_model
from video_io import iter_video_frames
import torch
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import base64

from sports.common.team import TeamClassifier


def collect_player_crops(
    video_path: str,
    detection_model,
    player_class_id: int = PLAYER_ID,
    stride: int = 30,
) -> List[np.ndarray]:
    """
    Collect cropped player images from a video.

    We:
    - Sample frames every `stride` frames.
    - Run detection.
    - Keep only `player_class_id` detections.
    - Crop them from the frame and accumulate.

    Args:
        video_path:      Path to the video file.
        detection_model: Roboflow detection model.
        player_class_id: Class ID allocated to players in the model.
        stride:          Frame stride for sampling.

    Returns:
        List of NumPy arrays (BGR) representing player crops.
    """
    crops: List[np.ndarray] = []

    for frame in tqdm(iter_video_frames(video_path, stride=stride), desc="collecting crops"):
        result = detection_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        player_detections = detections[detections.class_id == player_class_id]

        frame_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
        crops.extend(frame_crops)

    return crops


def compute_siglip_embeddings(
    crops: List[np.ndarray],
    device_str: str = "cuda",
) -> np.ndarray:
    """
    Compute SigLIP embeddings for a list of player crops.

    This is the manual approach for embedding extraction.

    Args:
        crops:     List of NumPy images (BGR).
        device_str:Preferred Torch device ("cuda" or "cpu").

    Returns:
        Numpy array of shape (N, D) where D is the embedding size (e.g. 768).
    """
    device = get_torch_device(device_str)
    model, processor = init_siglip_embedding_model(device=device_str)

    # Convert crops to PIL for the processor
    crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(
            chunked(crops_pil, EMBEDDING_BATCH_SIZE),
            desc="embedding extraction",
        ):
            inputs = processor(images=list(batch), return_tensors="pt").to(device)
            outputs = model(**inputs)

            # Mean pooling over sequence dimension
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    data = np.concatenate(all_embeddings, axis=0)
    return data


def cluster_embeddings(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
    n_clusters: int = KMEANS_NUM_CLUSTERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings with UMAP and cluster them with KMeans.

    Args:
        embeddings: (N, D) embedding matrix.
        n_components: Number of UMAP dimensions to keep.
        n_clusters:   Number of clusters for KMeans.

    Returns:
        projections: (N, n_components) UMAP projection.
        labels:      (N,) KMeans cluster labels.
    """
    reducer = umap.UMAP(n_components=n_components)
    projections = reducer.fit_transform(embeddings)

    clustering_model = KMeans(n_clusters=n_clusters, n_init="auto")
    labels = clustering_model.fit_predict(projections)

    return projections, labels


def _pil_to_data_uri(image: Image.Image) -> str:
    """
    Convert a PIL image to a base64 PNG data URI for HTML rendering.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def visualize_embeddings_3d(
    projections: np.ndarray,
    labels: np.ndarray,
    crops: List[np.ndarray],
    show_legend: bool = False,
) -> go.Figure:
    """
    Create a Plotly 3D scatter plot of embeddings.

    Each point represents one player crop; cluster labels are displayed.

    Args:
        projections: (N, 3) UMAP projections.
        labels:      (N,) cluster labels.
        crops:       List of crops (NumPy BGR images).
        show_legend: Whether to show the legend.

    Returns:
        Plotly Figure instance that you can `.show()` in a Python environment.
    """
    # Convert crops to PIL
    crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]

    # Prepare a mapping from index to image data URI
    image_data_uris = [ _pil_to_data_uri(img) for img in crops_pil ]

    unique_labels = np.unique(labels)
    traces = []

    for unique_label in unique_labels:
        mask = labels == unique_label

        traces.append(
            go.Scatter3d(
                x=projections[mask][:, 0],
                y=projections[mask][:, 1],
                z=projections[mask][:, 2],
                mode="markers",
                text=[str(unique_label)] * np.sum(mask),
                name=f"cluster_{unique_label}",
                marker=dict(size=4),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
        ),
        showlegend=show_legend,
    )

    # Note: if you want to attach on-click image previews,
    # you can extend this function with custom JavaScript / Dash.
    return fig


# ----------------------------------------------------------------------
# High-level team classifier (sports TeamClassifier)
# ----------------------------------------------------------------------


def build_team_classifier(
    video_path: str,
    detection_model,
    device: str = "cuda",
    stride: int = 30,
) -> TeamClassifier:
    """
    Build and fit a TeamClassifier using the sports library.

    This is the recommended way to split players into teams, since it wraps:
    - SigLIP embeddings
    - UMAP dimensionality reduction
    - KMeans clustering

    Args:
        video_path:      Path to the video used to collect training crops.
        detection_model: Roboflow detection model.
        device:          Device string ("cuda" or "cpu").
        stride:          Frame stride for sampling crops.

    Returns:
        Fitted TeamClassifier instance.
    """
    print("Collecting player crops for TeamClassifier...")
    crops = collect_player_crops(
        video_path=video_path,
        detection_model=detection_model,
        player_class_id=PLAYER_ID,
        stride=stride,
    )

    print(f"Collected {len(crops)} crops. Fitting TeamClassifier...")
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    print("TeamClassifier fitted successfully.")
    return team_classifier


def assign_player_teams(
    players_detections: sv.Detections,
    frame: np.ndarray,
    team_classifier: TeamClassifier,
) -> sv.Detections:
    """
    Assign team IDs (0 or 1) to player detections for a single frame.

    Args:
        players_detections: Detections corresponding to players only.
        frame:              Full frame image from which detections were croped.
        team_classifier:    Fitted TeamClassifier.

    Returns:
        Updated `players_detections` where class_id is now the team ID.
    """
    player_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    team_ids = team_classifier.predict(player_crops)
    players_detections.class_id = team_ids
    return players_detections


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    """
    Assign goalkeepers to a team based on their proximity to team centroids.

    Approach:
    - Compute the bottom-center anchor coordinates for each player.
    - Compute separate centroids for team 0 and team 1.
    - Assign each goalkeeper to whichever centroid is closer (in pixel space).

    Args:
        players:     Detections where `class_id` is the team ID (0 or 1).
        goalkeepers: Detections corresponding to goalkeepers.

    Returns:
        Numpy array of team IDs for each goalkeeper detection.
    """
    if len(goalkeepers) == 0 or len(players) == 0:
        return np.array([], dtype=int)

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Compute centroids for team 0 and team 1
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)

    keeper_team_ids: List[int] = []

    for keeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(keeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(keeper_xy - team_1_centroid)
        team_id = 0 if dist_0 < dist_1 else 1
        keeper_team_ids.append(team_id)

    return np.array(keeper_team_ids, dtype=int)
