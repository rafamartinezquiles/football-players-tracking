# Interpreting Player Dynamics with Computer Vision for Tactical Soccer Analysis
Using computer vision and machine learning techniques, including Roboflow Inference, Supervision, Torch, Transformers (SigLIP), OpenCV, UMAP, scikit-learn and NVIDIA’s GPU stack where available, to detect and track football players, referees, goalkeepers, and the ball, then project them onto a 2D tactical pitch view.
![](images/boxes_with_labels.png)

## Overview and Background
Analyzing football matches automatically involves several challenging subtasks: locating all players, referees and the ball in broadcast footage, deciding which team each player belongs to, understanding the geometry of the pitch from the camera view, and finally transforming everything into useful tactical information. In this project, we build a modular Football AI system that chains together object detection, multi-object tracking, team-color clustering and pitch keypoint detection to create a “video-game style” representation of a real match. Roboflow detection models are used to find balls, players, goalkeepers and referees frame-by-frame; Supervision and ByteTrack maintain consistent identities over time, while a dedicated field model extracts keypoints that define the pitch layout from the camera perspective.

On top of these detections, we leverage a SigLIP-based embedding model together with UMAP and KMeans (and a higher-level TeamClassifier wrapper) to cluster players by jersey color and automatically split them into two teams. Using homography (via a ViewTransformer) between the camera plane and an idealized soccer pitch, all entities are projected into metric pitch coordinates. This enables the creation of radar-style tactical views, Voronoi control maps, and smoothed ball trajectories in 2D, giving coaches and analysts an interpretable, data-driven view of positioning, space control and ball movement derived directly from raw match video.
![](images/radar_view.png)

## Overview and Background
Detecting and recognizing Racing Bib Numbers (RBN) entails the challenging tasks of locating the bib attached to a person within a natural scene and deciphering the text inscribed on the bib itself in order to identify the runner. This involves intricate steps such as identifying the bib's area on a person and then interpreting the numerical content. Drawing from prior research and practical experience, this project employs a functional Convolutional Neural Network (CNN) to effectively identify race bib numbers in static images.

This repository delves into the exploration of Convolutional Neural Networks (CNN), particularly focusing on the utilization of Ultralytics (in particular YOLO module) and OpenCV, to discern Racing Bib Numbers (RBNR) within natural image settings. By capitalizing on publicly available and labeled datasets sourced from earlier investigations (refer to the reference section for additional details), results have been achieved in terms of accuracy and prediction time.
![](images/results_accuracy.png)

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
