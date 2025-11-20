"""
pitch.py

Everything related to the football pitch:

- Detecting pitch keypoints using the Roboflow field model.
- Filtering keypoints by confidence.
- Computing a homography (ViewTransformer) between the frame and pitch plane.
- Projecting points (ball, players, referees) between frame and pitch.
- Drawing:
  - Abstract pitch,
  - Radar view (positions on pitch),
  - Voronoi control diagrams (standard and custom blended).

By isolating this logic, the rest of the code can simply call
high-level functions such as `project_entities_to_pitch` or
`draw_radar_view`.
"""

from typing import Tuple, Optional

import numpy as np
import supervision as sv
import cv2

from config import BALL_ID, HOMOGRAPHY_SMOOTHING_WINDOW
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
)
from sports.common.view import ViewTransformer


def detect_pitch_keypoints(
    frame,
    field_model,
    confidence: float = 0.3,
) -> sv.KeyPoints:
    """
    Detect pitch keypoints using the given Roboflow field model.

    Args:
        frame:       Input frame (NumPy, BGR).
        field_model: Roboflow keypoint model.
        confidence:  Confidence threshold.

    Returns:
        Supervision `KeyPoints` object.
    """
    result = field_model.infer(frame, confidence=confidence)[0]
    key_points = sv.KeyPoints.from_inference(result)
    return key_points


def filter_keypoints_by_confidence(
    key_points: sv.KeyPoints,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Filter pitch keypoints by confidence.

    Args:
        key_points: KeyPoints object.
        threshold:  Minimum confidence to keep.

    Returns:
        frame_reference_points: Numpy array of shape (K, 2) with high-confidence points.
    """
    mask = key_points.confidence[0] > threshold
    frame_reference_points = key_points.xy[0][mask]
    return frame_reference_points


def get_view_transformer_frame_to_pitch(
    frame_reference_points: np.ndarray,
    config: SoccerPitchConfiguration,
) -> ViewTransformer:
    """
    Build a ViewTransformer that maps from frame coordinates to pitch coordinates.

    The mapping is based on pairs of reference points:
    - `frame_reference_points` in the image plane.
    - Corresponding vertices from the `config`.

    Args:
        frame_reference_points: High-confidence keypoints in the frame.
        config:                 SoccerPitchConfiguration instance.

    Returns:
        ViewTransformer instance where source=frame, target=pitch.
    """
    pitch_reference_points = np.array(config.vertices)[ : frame_reference_points.shape[0] ]
    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points,
    )
    return transformer


def project_entities_to_pitch(
    ball_detections: sv.Detections,
    players_detections: sv.Detections,
    referees_detections: sv.Detections,
    transformer: ViewTransformer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project ball, players, and referees from frame coordinates to pitch coordinates.

    Uses the bottom-center anchor of each bounding box as the entity's position.

    Args:
        ball_detections:    Detections corresponding to the ball.
        players_detections: Detections corresponding to players/goalkeepers.
        referees_detections:Detections corresponding to referees.
        transformer:        Homography transformer mapping frame->pitch.

    Returns:
        pitch_ball_xy, pitch_players_xy, pitch_referees_xy
    """
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
    pitch_players_xy = transformer.transform_points(points=frame_players_xy)
    pitch_referees_xy = transformer.transform_points(points=frame_referees_xy)

    return pitch_ball_xy, pitch_players_xy, pitch_referees_xy


def draw_radar_view(
    config: SoccerPitchConfiguration,
    pitch_ball_xy: np.ndarray,
    pitch_players_xy: np.ndarray,
    players_team_ids: np.ndarray,
    pitch_referees_xy: np.ndarray,
) -> np.ndarray:
    """
    Draw a "radar view" of positions on the abstract pitch.

    - Ball is drawn as a white circle.
    - Team 0 as blue, team 1 as pink.
    - Referees as yellow.

    Args:
        config:            SoccerPitchConfiguration instance.
        pitch_ball_xy:     Ball positions in pitch coordinates (Kx2).
        pitch_players_xy:  Player/goalkeeper positions in pitch coords (Nx2).
        players_team_ids:  Team ID (0 or 1) for each row in `pitch_players_xy`.
        pitch_referees_xy: Referee positions in pitch coords (Rx2).

    Returns:
        Annotated pitch image as a NumPy array.
    """
    pitch_image = draw_pitch(config)

    # Draw ball
    pitch_image = draw_points_on_pitch(
        config=config,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=pitch_image,
    )

    # Team 0
    pitch_image = draw_points_on_pitch(
        config=config,
        xy=pitch_players_xy[players_team_ids == 0],
        face_color=sv.Color.from_hex("00BFFF"),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=pitch_image,
    )

    # Team 1
    pitch_image = draw_points_on_pitch(
        config=config,
        xy=pitch_players_xy[players_team_ids == 1],
        face_color=sv.Color.from_hex("FF1493"),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=pitch_image,
    )

    # Referees
    pitch_image = draw_points_on_pitch(
        config=config,
        xy=pitch_referees_xy,
        face_color=sv.Color.from_hex("FFD700"),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=pitch_image,
    )

    return pitch_image


def draw_custom_voronoi_blend(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.from_hex("00BFFF"),
    team_2_color: sv.Color = sv.Color.from_hex("FF1493"),
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw a smooth Voronoi-like control map for two teams on the pitch.

    This is a custom version of a Voronoi diagram that uses a tanh-based
    blending function to create soft transitions between team regions.

    Args:
        config:       Pitch configuration.
        team_1_xy:    Positions of team 1 players in pitch coordinates.
        team_2_xy:    Positions of team 2 players in pitch coordinates.
        team_1_color: Color for team 1 regions.
        team_2_color: Color for team 2 regions.
        opacity:      Opacity of the overlay.
        padding:      Padding in pixels around pitch.
        scale:        Scale factor from pitch units to pixels.
        pitch:        Optional base pitch image. If None, a new one is created.

    Returns:
        Image of pitch with colored control regions.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    color_1 = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    color_2 = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    # Grid of pixel coordinates
    y_coords, x_coords = np.indices(
        (scaled_width + 2 * padding, scaled_length + 2 * padding)
    )
    y_coords -= padding
    x_coords -= padding

    def compute_distances(xy: np.ndarray) -> np.ndarray:
        """
        Compute distances from each team point to every pixel in the grid.
        """
        return np.sqrt(
            (xy[:, 0][:, None, None] * scale - x_coords) ** 2
            + (xy[:, 1][:, None, None] * scale - y_coords) ** 2
        )

    distances_team_1 = compute_distances(team_1_xy)
    distances_team_2 = compute_distances(team_2_xy)

    min_dist_1 = np.min(distances_team_1, axis=0)
    min_dist_2 = np.min(distances_team_2, axis=0)

    # Avoid division by zero
    denom = np.clip(min_dist_1 + min_dist_2, a_min=1e-5, a_max=None)
    distance_ratio = min_dist_2 / denom

    # Tanh-based blend for sharper transitions
    steepness = 15.0
    blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

    # Interpolate between colors for each channel
    for c in range(3):
        voronoi[:, :, c] = (
            blend_factor * color_1[c] + (1.0 - blend_factor) * color_2[c]
        ).astype(np.uint8)

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1.0 - opacity, 0.0)
    return overlay
