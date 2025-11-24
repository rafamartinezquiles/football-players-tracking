# Interpreting Player Dynamics with Computer Vision for Tactical Soccer Analysis
Using computer vision and machine learning techniques, including Roboflow Inference, Supervision, Torch, Transformers (SigLIP), OpenCV, UMAP, scikit-learn and NVIDIA’s GPU stack where available, to detect and track football players, referees, goalkeepers, and the ball, then project them onto a 2D tactical pitch view.

## Overview and Background
Analyzing football matches automatically involves several challenging subtasks: locating all players, referees and the ball in broadcast footage, deciding which team each player belongs to, understanding the geometry of the pitch from the camera view, and finally transforming everything into useful tactical information. In this project, we build a modular Football AI system that chains together object detection, multi-object tracking, team-color clustering and pitch keypoint detection to create a “video-game style” representation of a real match. Roboflow detection models are used to find balls, players, goalkeepers and referees frame-by-frame; Supervision and ByteTrack maintain consistent identities over time, while a dedicated field model extracts keypoints that define the pitch layout from the camera perspective.

On top of these detections, we leverage a SigLIP-based embedding model together with UMAP and KMeans (and a higher-level TeamClassifier wrapper) to cluster players by jersey color and automatically split them into two teams. Using homography (via a ViewTransformer) between the camera plane and an idealized soccer pitch, all entities are projected into metric pitch coordinates. This enables the creation of radar-style tactical views, Voronoi control maps, and smoothed ball trajectories in 2D, giving coaches and analysts an interpretable, data-driven view of positioning, space control and ball movement derived directly from raw match video.

## Table of Contents
```
football-players-tracking
|__ images
|   |__ ball_path.png 
|   |__ boxes_with_labels.png
|   |__ radar_view.png
|   |__ stylized_view.png
|   |__ tracked_view.png
|   |__ voronoi_basic.png
|   |__ voronoi_blend.png
|__ src
    |__ __init__.py
    |__ ball_tracking.py
    |__ config.py
    |__ detection.py
    |__ models.py
    |__ pitch.py
    |__ project_secrets.py
    |__ run_demos.py
    |__ team_clustering.py
    |__ tracking.py
    |__ video_io.py
|__ videos
|   |__ 0bfacc_0.mp4
|   |__ 2e57b9_0.mp4
|   |__ 08fd33_0.mp4
|   |__ 573e61_0.mp4
|   |__ 121364_0.mp4
README.md
requirements.txt
LICENSE
```

## Getting started

### Resources used
A high-performance Acer Nitro 5 laptop, powered by an Intel Core i7 processor and an NVIDIA GeForce GTX 1650 GPU (4 GB VRAM), was used for model training and evaluation. Due to the large size of the dataset, the training process was computationally demanding and prolonged. Nevertheless, this hardware configuration provided a stable and efficient environment, enabling consistent experimentation and reliable validation of the gesture-recognition models.

### Installing
The project is deployed in a local machine, so you need to install the next software and dependencies to start working:

1. Create and activate the new virtual environment for the project

```bash
conda create --name football_ai python=3.11
conda activate football_ai
```

2. Clone repository

```bash
git clone https://github.com/rafamartinezquiles/football-players-tracking.git
```

3. In the same folder that the requirements are, install the necessary requirements

```bash
cd football-players-tracking
pip install -r requirements.txt
```

4. In addition to installing the required packages, you must set up the API keys needed to access HuggingFace models and Roboflow. Create an account on each platform and generate your API keys. Note: for HuggingFace, a read-only token is sufficient. Once you have both keys, set them in your environment. On Windows (Command Prompt), run:

```bash
set HF_TOKEN=api_key
set ROBOFLOW_API_KEY=api_key
```

### Execution
From the project root (folder containing src/), execute:

```bash
python src/run_demos.py
```

The script automatically executes three demos in sequence: single-frame detection, radar and Voronoi analysis, and ball trajectory extraction. If needed, any of these demos can be disabled by commenting them out at the bottom of run_demos.py, allowing you to run them individually during debugging or development.

During execution, the first stage—single-frame detection—starts with a call to init_roboflow_models(), which loads both the player/ball/referee detection model and the pitch keypoint model. The system then reads the first video frame from SOURCE_VIDEO_PATH using Supervision’s efficient frame generator. That frame is passed to detection.run_player_detection_on_frame, which performs inference and converts the raw Roboflow response into a standardized Detections object. Annotation utilities such as create_box_and_label_annotators and annotate_frame_with_boxes overlay bounding boxes and class/confidence labels on detected players, goalkeepers, referees and the ball. A second visualization, generated by stylized_frame_view, provides a “video-game style” representation, replacing bounding boxes with colored ellipses and a triangular pointer for the ball. When the stage completes, two images appear in your project root: boxes_with_labels.png, showing the standard annotated frame, and stylized_view.png, showing the stylized version.

The second stage—team clustering, radar view and Voronoi maps—begins in demo_radar_and_voronoi() by building a TeamClassifier. This process samples video frames at a configurable stride (default: every 30 frames), detects players, crops them, and feeds the crops into SigLIP to obtain high-dimensional visual embeddings. These embeddings are reduced with UMAP and clustered with KMeans, creating a model capable of distinguishing jersey colors and therefore assigning players to teams. A representative frame is then processed again: the detector runs, the ball is isolated and padded, and all remaining detections pass through NMS before being fed into a ByteTrack instance (tracking.create_byte_tracker) to produce stable tracker_id assignments. Detections are separated into players, goalkeepers and referees.

Team assignment follows. assign_player_teams classifies players using the trained TeamClassifier, while resolve_goalkeepers_team_id computes each team’s centroid on the pitch and assigns each goalkeeper to the nearest team. The frame is then rendered using annotate_tracked_frame, which draws the stylized ellipses, tracker IDs, and ball marker, and is saved as tracked_view.png.

Next, the system extracts pitch geometry. detect_pitch_keypoints runs the Roboflow field-keypoints model, and filter_keypoints_by_confidence removes unreliable points. The remaining keypoints are matched with known pitch vertices from SoccerPitchConfiguration, and get_view_transformer_frame_to_pitch computes a homography mapping the broadcast camera view onto normalized pitch coordinates. Using this mapping, project_entities_to_pitch transforms the ball, players and referees into 2D pitch positions. draw_radar_view then generates a simplified tactical “radar” display: the ball is shown as a white dot, players of team 0 in blue, players of team 1 in pink, and referees in yellow. This radar visualization is saved as radar_view.png.

Finally, the system creates two Voronoi control maps. A base pitch is drawn with draw_pitch, after which draw_pitch_voronoi_diagram colors each region of the pitch according to the nearest player, producing a classical control map saved as voronoi_basic.png. A more advanced, smooth-blended version, produced by draw_custom_voronoi_blend in pitch.py, uses a tanh-based transition between colors and is saved as voronoi_blend.png. When this stage ends, you have a consistent mapping from the broadcast frame to tactical pitch space and two complete styles of spatial control visualization.

The third demo, ball trajectory extraction, processes the entire video frame by frame. demo_ball_path() initializes the models and loads a SoccerPitchConfiguration before calling ball_tracking.compute_ball_path_raw. For each frame, the system detects the ball and the field keypoints, and attempts to compute a valid camera-to-pitch homography. If reliable keypoints are available, the ball’s bottom-center location is mapped into pitch coordinates and added to the trajectory list. A deque of recent homographies is maintained to smooth jitter and maintain stable projections. After the full raw trajectory is collected, replace_outliers_based_on_distance removes unrealistic jumps by comparing each position to the last valid one. The cleaned path is then rendered using draw_ball_path_on_pitch, producing ball_path.png, which shows the ball’s movement over the entire match. Since this process iterates through every frame of the video, it is typically the slowest step, especially on CPU.

After a complete run, your project root should contain: boxes_with_labels.png, stylized_view.png, tracked_view.png, radar_view.png, voronoi_basic.png, voronoi_blend.png, and ball_path.png. Viewed in sequence, these images show the detections, stylized visualizations, per-player tracking, radar projection, team-control maps, and final ball trajectory. If any errors occur during execution, you can isolate the problem by running only one demo at a time in run_demos.py and using the traceback to diagnose the issue.