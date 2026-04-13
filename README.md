# Batch Pose Estimation with IMU and Stereo Measurements

This repository implements a nonlinear batch state estimation pipeline for 3D pose estimation using IMU and stereo measurements.

## Overview
The project estimates the pose trajectory of a moving sensor platform by combining motion information from IMU measurements with stereo landmark observations. The estimator is formulated as a nonlinear least-squares problem and solved iteratively.

## Method
The implementation includes:
- SE(3) pose representation
- IMU-based motion integration
- Stereo landmark reprojection residuals
- Batch nonlinear least-squares optimization
- Levenberg-Marquardt style damping
- Sliding-window estimation experiments
- Position and rotation error visualization

## Files
- `Batch_Pose_imu_stereo.py`: main implementation
- `requirements.txt`: Python dependencies
- `.gitignore`: excludes datasets, generated figures, and cache files

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run
This script requires the dataset file `dataset3.mat`, which is not included in this repository.
After preparing the dataset and updating the dataset path inside the script, run:

```bash
python Batch_Pose_imu_stereo.py
```

## Notes
This repository is adapted from a graduate state estimation course project at the University of Toronto.
The dataset and original course handout are not included in this repository.
