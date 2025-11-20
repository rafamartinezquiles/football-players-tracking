"""
tracking.py

This module handles player tracking using ByteTrack from the Supervision library.

Responsibilities:
- Creating and resetting a ByteTrack instance.
- Updating the tracker with the detections for a new frame.
- Building human-readable labels based on tracker IDs.
- Annotating a frame with tracked IDs using Ellipse and Label annotators.

Tracking is separated from detection to keep responsibilities clear.
"""

from typing import List, Tuple

import numpy as np
import supervision as sv


def create_byte_tracker() -> sv.ByteTrack:
    """
    Create a new ByteTrack instance.

    Returns:
        A fresh `sv.ByteTrack` tracker with default settings.
    """
    tracker = sv.ByteTrack()
    tracker.reset()
    return tracker


def update_tracker(
    tracker: sv.ByteTrack,
    detections: sv.Detections,
) -> sv.Detections:
    """
    Update the tracker with the current frame's detections.

    Args:
        tracker:    ByteTrack instance.
        detections: Detections for the current frame.

    Returns:
        Updated detections that now include `tracker_id` for each detection.
    """
    updated_detections = tracker.update_with_detections(detections=detections)
    return updated_detections


def build_tracker_labels(detections: sv.Detections) -> List[str]:
    """
    Build textual labels for each detection based on its tracker ID.

    Args:
        detections: Detections updated by ByteTrack.

    Returns:
        List of labels such as "#1", "#2", ...
    """
    return [f"#{tracker_id}" for tracker_id in detections.tracker_id]


def create_tracking_annotators() -> Tuple[sv.EllipseAnnotator, sv.LabelAnnotator]:
    """
    Create annotators used for visualizing tracked entities.

    Returns:
        ellipse_annotator, label_annotator
    """
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER,
    )
    return ellipse_annotator, label_annotator


def annotate_tracked_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    ball_detections: sv.Detections,
) -> np.ndarray:
    """
    Annotate a single frame with tracked entities and a highlighted ball.

    Args:
        frame:          Original frame.
        detections:     All non-ball detections (with tracker_id).
        ball_detections:Ball detections (should already be padded).

    Returns:
        Annotated frame.
    """
    ellipse_annotator, label_annotator = create_tracking_annotators()
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=25,
        height=21,
        outline_thickness=1,
    )

    labels = build_tracker_labels(detections)
    annotated_frame = frame.copy()

    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
    )
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame,
        detections=ball_detections,
    )
    return annotated_frame
