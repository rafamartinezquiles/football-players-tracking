"""
detection.py

Functions related to object detection in a single frame:

- Running the Roboflow model on a frame.
- Splitting detections into ball vs other entities.
- Producing standard annotated views:
  - Bounding boxes with labels.
  - Stylized "video game style" view (ellipses and triangle).

We operate on Supervision's `Detections` structure, which provides
convenient slicing, NMS, and coordinate access.
"""

from typing import Tuple, List

import numpy as np
import supervision as sv

from config import BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID


def run_player_detection_on_frame(
    frame: np.ndarray,
    model,
    confidence: float = 0.3,
) -> sv.Detections:
    """
    Run Roboflow player/ball/referee detection on a single frame.

    Args:
        frame:      Input frame (NumPy array, BGR).
        model:      Roboflow model returned by `inference.get_model`.
        confidence: Confidence threshold for predictions.

    Returns:
        Supervision `Detections` object containing all detections.
    """
    result = model.infer(frame, confidence=confidence)[0]
    detections = sv.Detections.from_inference(result)
    return detections


def split_detections_by_role(
    detections: sv.Detections,
) -> Tuple[sv.Detections, sv.Detections, sv.Detections, sv.Detections]:
    """
    Split detections into ball, goalkeepers, players, and referees.

    Args:
        detections: `sv.Detections` containing all types.

    Returns:
        ball_detections, goalkeeper_detections,
        player_detections, referee_detections
    """
    ball = detections[detections.class_id == BALL_ID]
    goalkeepers = detections[detections.class_id == GOALKEEPER_ID]
    players = detections[detections.class_id == PLAYER_ID]
    referees = detections[detections.class_id == REFEREE_ID]
    return ball, goalkeepers, players, referees


def create_box_and_label_annotators() -> Tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    """
    Construct box and label annotators with a fixed color palette.

    The palette maps class IDs to specific colors for easier debugging.

    Returns:
        box_annotator, label_annotator
    """
    palette = sv.ColorPalette.from_hex(["#FF8C00", "#00BFFF", "#FF1493", "#FFD700"])
    box_annotator = sv.BoxAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=palette, text_color=sv.Color.from_hex("#000000")
    )
    return box_annotator, label_annotator


def annotate_frame_with_boxes(
    frame: np.ndarray,
    detections: sv.Detections,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
) -> np.ndarray:
    """
    Draw bounding boxes and labels for each detection on a copy of the input frame.

    Args:
        frame:           Original frame.
        detections:      Supervision detections.
        box_annotator:   Box annotator instance.
        label_annotator: Label annotator instance.

    Returns:
        Annotated frame as NumPy array.
    """
    # Build labels as "<class_name> <confidence>"
    labels: List[str] = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections["class_name"], detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
    )

    return annotated_frame


def create_stylized_annotators():
    """
    Create annotators for the "video game style" view:

    - EllipseAnnotator: draws ellipses for non-ball actors
    - TriangleAnnotator: draws a triangle marker for the ball

    Returns:
        ellipse_annotator, triangle_annotator
    """
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        thickness=2,
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=25,
        height=21,
        outline_thickness=1,
    )
    return ellipse_annotator, triangle_annotator


def stylized_frame_view(
    frame: np.ndarray,
    detections: sv.Detections,
    ball_id: int = BALL_ID,
    nms_threshold: float = 0.5,
) -> np.ndarray:
    """
    Produce a stylized "video game" view with ellipses and a triangle.

    Pipeline:
    - Separate ball detections from others.
    - Pad the ball bounding box slightly.
    - Apply NMS on non-ball detections.
    - Shift class IDs (for palette indexing).
    - Draw ellipses for non-ball actors and a triangle for the ball.

    Args:
        frame:          Original frame.
        detections:     All detections.
        ball_id:        Class ID for the ball.
        nms_threshold:  IoU threshold for class-agnostic NMS.

    Returns:
        Annotated frame.
    """
    ellipse_annotator, triangle_annotator = create_stylized_annotators()

    # Separate ball detections
    ball_detections = detections[detections.class_id == ball_id]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # Non-ball detections
    all_detections = detections[detections.class_id != ball_id]
    all_detections = all_detections.with_nms(threshold=nms_threshold, class_agnostic=True)

    # Rebase class IDs for consistent styling (optional but convenient)
    all_detections.class_id -= 1

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections,
    )
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame,
        detections=ball_detections,
    )

    return annotated_frame
