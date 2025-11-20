"""
ball_tracking.py

Implements ball tracking in pitch coordinates across an entire video.

Key features:
- For each frame, detects:
    - Ball bounding box.
    - Pitch keypoints.
- Builds a homography (frame → pitch) for each frame using ViewTransformer.
- Smooths the homography over a sliding window of frames.
- Projects the ball position onto the pitch at each frame.
- Cleans the track by removing outlier jumps.
- Draws the resulting path on the abstract pitch.

This module is focused solely on the time dimension and path logic.
"""

from collections import deque
from typing import List

import numpy as np
import supervision as sv
from tqdm import tqdm

from config import (
    BALL_ID,
    HOMOGRAPHY_SMOOTHING_WINDOW,
    MAX_BALL_DISTANCE_THRESHOLD,
)
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_paths_on_pitch
from pitch import detect_pitch_keypoints, filter_keypoints_by_confidence
from video_io import get_video_info, iter_video_frames


def compute_ball_path_raw(
    video_path: str,
    detection_model,
    field_model,
    config: SoccerPitchConfiguration,
    confidence: float = 0.3,
) -> List[np.ndarray]:
    """
    Compute the raw ball path in pitch coordinates across the entire video.

    For each frame:
    - Detect the ball.
    - Detect pitch keypoints.
    - Compute homography frame->pitch.
    - Smooth the homography over the last M frames.
    - Project the ball bottom-center to pitch coordinates.

    Args:
        video_path:      Path to the video file.
        detection_model: Roboflow detection model for ball.
        field_model:     Roboflow field keypoint model.
        config:          SoccerPitchConfiguration.
        confidence:      Detection confidence threshold.

    Returns:
        List of arrays with shape (1, 2) or (0, 2) for each frame.
    """
    video_info = get_video_info(video_path)
    frames = iter_video_frames(video_path)

    homography_history = deque(maxlen=HOMOGRAPHY_SMOOTHING_WINDOW)
    path_raw: List[np.ndarray] = []

    for frame in tqdm(frames, total=video_info.total_frames, desc="ball tracking"):
        # Detect ball
        result = detection_model.infer(frame, confidence=confidence)[0]
        detections = sv.Detections.from_inference(result)
        ball_detections = detections[detections.class_id == BALL_ID]

        # If no ball detected in this frame, record an empty entry and continue
        if len(ball_detections) == 0:
            path_raw.append(np.empty((0, 2), dtype=np.float32))
            continue

        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # Detect pitch keypoints
        key_points = detect_pitch_keypoints(
            frame=frame,
            field_model=field_model,
            confidence=confidence,
        )
        frame_reference_points = filter_keypoints_by_confidence(key_points, threshold=0.5)

        if frame_reference_points.shape[0] < 4:
            # Not enough keypoints to compute a reliable homography
            path_raw.append(np.empty((0, 2), dtype=np.float32))
            continue

        # Build a homography from frame to pitch
        pitch_reference_points = np.array(config.vertices)[: frame_reference_points.shape[0]]
        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points,
        )

        homography_history.append(transformer.m)
        transformer.m = np.mean(np.array(homography_history), axis=0)

        # Project ball bottom-center to pitch
        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

        path_raw.append(pitch_ball_xy)

    return path_raw


def replace_outliers_based_on_distance(
    positions: List[np.ndarray],
    distance_threshold: float,
) -> List[np.ndarray]:
    """
    Remove outliers from a sequence of positions based on jump distance.

    If the distance from the last valid position exceeds `distance_threshold`,
    the position is replaced with an empty array.

    Args:
        positions:          List of positions per frame, each (N, 2) or empty.
        distance_threshold: Maximum allowed step size between frames.

    Returns:
        Cleaned list of positions.
    """
    cleaned: List[np.ndarray] = []
    last_valid: np.ndarray | None = None

    for pos in positions:
        if len(pos) == 0:
            # No detection in this frame
            cleaned.append(pos)
            continue

        if last_valid is None:
            cleaned.append(pos)
            last_valid = pos
            continue

        distance = np.linalg.norm(pos - last_valid)
        if distance > distance_threshold:
            # Too big a jump, treat as outlier
            cleaned.append(np.empty((0, 2), dtype=np.float32))
        else:
            cleaned.append(pos)
            last_valid = pos

    return cleaned


def draw_ball_path_on_pitch(
    config: SoccerPitchConfiguration,
    path: List[np.ndarray],
) -> np.ndarray:
    """
    Draw the ball path on the abstract pitch.

    Args:
        config: SoccerPitchConfiguration.
        path:   List of positions (per frame) in pitch coordinates.

    Returns:
        Annotated pitch image with the ball trajectory.
    """
    # Convert each 2D array into flattened (x, y) for the draw_paths_on_pitch API
    flattened = [coords.flatten() for coords in path]

    pitch_image = draw_pitch(config)
    pitch_image = draw_paths_on_pitch(
        config=config,
        paths=[flattened],
        color=sv.Color.WHITE,
        pitch=pitch_image,
    )
    return pitch_image


def compute_and_draw_clean_ball_path(
    video_path: str,
    detection_model,
    field_model,
    config: SoccerPitchConfiguration,
) -> np.ndarray:
    """
    Convenience wrapper:

    - Computes raw ball path.
    - Removes outliers using MAX_BALL_DISTANCE_THRESHOLD.
    - Draws the cleaned path on pitch.

    Args:
        video_path:      Path to video.
        detection_model: Roboflow detection model.
        field_model:     Roboflow field model.
        config:          SoccerPitchConfiguration.

    Returns:
        Pitch image with the ball trajectory.
    """
    raw_path = compute_ball_path_raw(
        video_path=video_path,
        detection_model=detection_model,
        field_model=field_model,
        config=config,
    )
    clean_path = replace_outliers_based_on_distance(
        raw_path,
        distance_threshold=MAX_BALL_DISTANCE_THRESHOLD,
    )
    pitch_image = draw_ball_path_on_pitch(config, clean_path)
    return pitch_image
