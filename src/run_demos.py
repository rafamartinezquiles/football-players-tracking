"""
run_demos.py

Entry-point script with a few demo functions showing how to use the modules:

1) demo_single_frame_detection:
   - Runs detection on a single frame and saves:
       * boxes_with_labels.png
       * stylized_view.png
       * tracked_view.png

2) demo_radar_and_voronoi:
   - Runs a single frame pipeline:
       * detect and track players
       * assign teams
       * detect pitch keypoints
       * project entities to pitch
       * draw radar view
       * draw Voronoi diagrams
   - Saves:
       * radar_view.png
       * voronoi_basic.png
       * voronoi_blend.png

3) demo_ball_path:
   - Tracks the ball over the entire video in pitch space.
   - Draws the trajectory and saves:
       * ball_path.png

You can run this script directly from the `src` directory:

    python run_demos.py

Make sure:
- Dependencies are installed.
- HF_TOKEN and ROBOFLOW_API_KEY environment variables are set.
- Sample videos are present at paths configured in config.py.
"""

import cv2
import supervision as sv

from config import SOURCE_VIDEO_PATH
from models import init_roboflow_models
from detection import (
    run_player_detection_on_frame,
    create_box_and_label_annotators,
    annotate_frame_with_boxes,
    stylized_frame_view,
)
from tracking import (
    create_byte_tracker,
    update_tracker,
    annotate_tracked_frame,
)
from team_clustering import (
    build_team_classifier,
    assign_player_teams,
    resolve_goalkeepers_team_id,
)
from pitch import (
    detect_pitch_keypoints,
    filter_keypoints_by_confidence,
    get_view_transformer_frame_to_pitch,
    project_entities_to_pitch,
    draw_radar_view,
    draw_custom_voronoi_blend,
)
from ball_tracking import compute_and_draw_clean_ball_path
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram
)


def demo_single_frame_detection():
    """
    Demo 1: Single-frame detection and visualization.

    Steps:
    - Load models.
    - Read a single frame from the source video.
    - Run detection.
    - Draw:
        * bounding boxes + labels
        * stylized view (ellipses + triangle for ball)
    - Save the images to disk.
    """
    print("Running demo_single_frame_detection...")

    player_model, _ = init_roboflow_models()

    # Read one frame from the video
    frame = next(sv.get_video_frames_generator(SOURCE_VIDEO_PATH))

    # Detection
    detections = run_player_detection_on_frame(frame, model=player_model, confidence=0.3)

    # Box + label view
    box_annotator, label_annotator = create_box_and_label_annotators()
    box_frame = annotate_frame_with_boxes(
        frame=frame,
        detections=detections,
        box_annotator=box_annotator,
        label_annotator=label_annotator,
    )
    cv2.imwrite("boxes_with_labels.png", box_frame)
    print("Saved boxes_with_labels.png")

    # Stylized view
    stylized = stylized_frame_view(frame=frame, detections=detections)
    cv2.imwrite("stylized_view.png", stylized)
    print("Saved stylized_view.png")

    print("demo_single_frame_detection complete.")


def demo_radar_and_voronoi():
    """
    Demo 2: Radar and Voronoi view for one representative frame.

    Steps:
    - Initialize models and team classifier.
    - Read one frame from the source video.
    - Detect and track players.
    - Assign teams to players and goalkeepers.
    - Detect pitch keypoints and build frame->pitch homography.
    - Project ball, players, referees to pitch coordinates.
    - Draw:
        * radar view
        * basic Voronoi
        * blended Voronoi
    - Save images to disk.
    """
    print("Running demo_radar_and_voronoi...")

    player_model, field_model = init_roboflow_models()
    config = SoccerPitchConfiguration()

    # Build TeamClassifier using the same video
    team_classifier = build_team_classifier(
        video_path=SOURCE_VIDEO_PATH,
        detection_model=player_model,
        device="cpu",
        stride=30,
    )

    # Read one frame
    frame = next(sv.get_video_frames_generator(SOURCE_VIDEO_PATH))

    # Detect entities
    detections = run_player_detection_on_frame(frame, model=player_model, confidence=0.3)

    # Separate ball and non-ball
    from config import BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID

    ball = detections[detections.class_id == BALL_ID]
    ball.xyxy = sv.pad_boxes(xyxy=ball.xyxy, px=10)

    others = detections[detections.class_id != BALL_ID]
    others = others.with_nms(threshold=0.5, class_agnostic=True)

    # Tracking
    tracker = create_byte_tracker()
    others_tracked = update_tracker(tracker, others)

    goalkeepers = others_tracked[others_tracked.class_id == GOALKEEPER_ID]
    players = others_tracked[others_tracked.class_id == PLAYER_ID]
    referees = others_tracked[others_tracked.class_id == REFEREE_ID]

    # Assign teams to players using TeamClassifier
    players = assign_player_teams(players_detections=players, frame=frame, team_classifier=team_classifier)

    # Assign teams to goalkeepers using heuristic
    keeper_team_ids = resolve_goalkeepers_team_id(players, goalkeepers)
    goalkeepers.class_id = keeper_team_ids

    # Map referee class IDs into "team-like" index space (optional, for consistent styling)
    referees.class_id -= 1

    # Merge players + goalkeepers + referees for frame visualization
    all_for_frame = sv.Detections.merge([players, goalkeepers, referees])

    # Visualize this frame with tracking overlay
    tracked_frame = annotate_tracked_frame(frame=frame, detections=all_for_frame, ball_detections=ball)
    cv2.imwrite("tracked_view.png", tracked_frame)
    print("Saved tracked_view.png")

    # Pitch keypoints and homography
    key_points = detect_pitch_keypoints(frame=frame, field_model=field_model, confidence=0.3)
    frame_ref_points, vertex_mask = filter_keypoints_by_confidence(key_points, threshold=0.5)

    if frame_ref_points.shape[0] < 4:
        raise RuntimeError("Not enough pitch keypoints for homography.")

    transformer = get_view_transformer_frame_to_pitch(
        frame_reference_points=frame_ref_points,
        vertex_mask=vertex_mask,
        config=config,
    )

    # Project all entities to pitch
    pitch_ball_xy, pitch_players_xy, pitch_referees_xy = project_entities_to_pitch(
        ball_detections=ball,
        players_detections=sv.Detections.merge([players, goalkeepers]),
        referees_detections=referees,
        transformer=transformer,
    )

    # Radar view
    radar = draw_radar_view(
        config=config,
        pitch_ball_xy=pitch_ball_xy,
        pitch_players_xy=pitch_players_xy,
        players_team_ids=sv.Detections.merge([players, goalkeepers]).class_id,
        pitch_referees_xy=pitch_referees_xy,
    )
    cv2.imwrite("radar_view.png", radar)
    print("Saved radar_view.png")

    # Basic Voronoi
    team_positions = sv.Detections.merge([players, goalkeepers])
    voronoi_basic = draw_pitch(config)
    voronoi_basic = draw_pitch_voronoi_diagram(
        config=config,
        team_1_xy=pitch_players_xy[team_positions.class_id == 0],
        team_2_xy=pitch_players_xy[team_positions.class_id == 1],
        team_1_color=sv.Color.from_hex("00BFFF"),
        team_2_color=sv.Color.from_hex("FF1493"),
        pitch=voronoi_basic,
    )
    cv2.imwrite("voronoi_basic.png", voronoi_basic)
    print("Saved voronoi_basic.png")

    # Blended Voronoi
    blended_base = draw_pitch(
        config=config,
        background_color=sv.Color.WHITE,
        line_color=sv.Color.BLACK,
    )
    voronoi_blend = draw_custom_voronoi_blend(
        config=config,
        team_1_xy=pitch_players_xy[team_positions.class_id == 0],
        team_2_xy=pitch_players_xy[team_positions.class_id == 1],
        team_1_color=sv.Color.from_hex("00BFFF"),
        team_2_color=sv.Color.from_hex("FF1493"),
        pitch=blended_base,
    )
    cv2.imwrite("voronoi_blend.png", voronoi_blend)
    print("Saved voronoi_blend.png")

    print("demo_radar_and_voronoi complete.")


def demo_ball_path():
    """
    Demo 3: Full ball trajectory in pitch space.

    Steps:
    - Initialize models.
    - Compute and clean ball path across all frames.
    - Draw the trajectory on the abstract pitch.
    - Save the result to disk.
    """
    print("Running demo_ball_path...")

    player_model, field_model = init_roboflow_models()
    config = SoccerPitchConfiguration()

    pitch_with_path = compute_and_draw_clean_ball_path(
        video_path=SOURCE_VIDEO_PATH,
        detection_model=player_model,
        field_model=field_model,
        config=config,
    )

    cv2.imwrite("ball_path.png", pitch_with_path)
    print("Saved ball_path.png")
    print("demo_ball_path complete.")


if __name__ == "__main__":
    # You can comment out demos you do not want to run.
    demo_single_frame_detection()
    demo_radar_and_voronoi()
    demo_ball_path()
