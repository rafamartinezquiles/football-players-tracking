"""
video_io.py

Utility functions for video reading and basic introspection.

We rely on the `supervision` library for:
- `VideoInfo`   (metadata such as total frames, fps, etc.)
- `get_video_frames_generator` (lazy frame-by-frame reading).

This module wraps those utilities so the rest of the code does not need to
interact with raw video APIs directly.
"""

from typing import Generator

import supervision as sv
import numpy as np


def get_video_info(path: str) -> sv.VideoInfo:
    """
    Load basic information about a video file.

    Args:
        path: Path to the video file on disk.

    Returns:
        sv.VideoInfo instance, which includes properties such as:
        - width, height
        - fps
        - total_frames
    """
    return sv.VideoInfo.from_video_path(path)


def iter_video_frames(
    path: str,
    stride: int = 1,
    start: int = 0,
) -> Generator[np.ndarray, None, None]:
    """
    Iterate over frames in a video with optional stride and start offset.

    Args:
        path:   Path to the video file.
        stride: Process every `stride`-th frame (default: 1 = every frame).
        start:  Index of the first frame to start from.

    Returns:
        Generator of frames as NumPy arrays (BGR, shape: H x W x 3).
    """
    frame_generator = sv.get_video_frames_generator(
        source_path=path,
        stride=stride,
        start=start,
    )
    for frame in frame_generator:
        yield frame
