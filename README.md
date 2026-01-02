# 3D LiDAR Change Detection

3D change detection between two LiDAR point cloud scenes using Open3D, ICP registration, and DBSCAN clustering.
Detect changed objects (like parked cars) by comparing two temporal point cloud captures of the same urban street scene. This project includes synthetic LiDAR data generation for testing and demonstration.

## Workflow


* Synthetic LiDAR Data Generation - Create realistic street-level point clouds

* Two-Scene Capture - Generate same scene at different timestamps with changes

* Point Cloud Registration - ICP (Iterative Closest Point) alignment for accurate matching

* Distance-Based Change Detection - KDTree spatial search for rapid change identification

* DBSCAN Clustering - Filters noise and identifies significant changes

* Ultra-HD Visualization - 2560×1440 interactive 3D viewer with color-coded results

* Automatic Report Export - Generates detailed statistics and analysis

* Optimized Processing - Voxel downsampling for faster computation


## Setup
 1. Create environment and activate
    
         python -m venv open3d_env
         source pen3d_env/bin/activate  # On Windows: pen3d_env\Scripts\activate
    
 3. Clone repository
