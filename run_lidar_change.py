import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import time

# ============= CONFIGURATION =============
BASE_DIR = "/home/abinandha/3d_pc_change/3DCDNet"
SAVE_DIR = os.path.join(BASE_DIR, "depth_captures")
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_change_detection_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ML Parameters
VOXEL_SIZE = 0.08
FEATURE_RADIUS = 0.5  # Radius for feature extraction
ANOMALY_CONTAMINATION = 0.12  # Expected proportion of outliers (increased to detect more)

print("="*100)
print(" "*20 + "MACHINE LEARNING 3D CHANGE DETECTION")
print(" "*15 + "Random Forest + Isolation Forest + Feature Engineering")
print("="*100)

# ============= LOAD POINT CLOUDS =============
def load_point_clouds():
    """Load both point cloud scenes."""
    print("\n[STEP 1] LOADING POINT CLOUDS...")
    
    scene1_path = os.path.join(SAVE_DIR, "ultimate_scene_1_250m_detailed.ply")
    scene2_path = os.path.join(SAVE_DIR, "ultimate_scene_2_250m_3people.ply")
    
    if not os.path.exists(scene1_path) or not os.path.exists(scene2_path):
        print(f"  ✗ ERROR: Point cloud files not found!")
        exit(1)
    
    print(f"  Loading Scene 1...")
    pcd1 = o3d.io.read_point_cloud(scene1_path)
    
    print(f"  Loading Scene 2...")
    pcd2 = o3d.io.read_point_cloud(scene2_path)
    
    print(f"  ✓ Scene 1: {len(pcd1.points):,} points")
    print(f"  ✓ Scene 2: {len(pcd2.points):,} points")
    
    return pcd1, pcd2

# ============= VOXEL DOWNSAMPLING =============
def downsample_point_cloud(pcd, voxel_size):
    """Downsample point cloud for faster processing."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

# ============= ICP REGISTRATION =============
def register_point_clouds(pcd1, pcd2):
    """Register Scene 2 to Scene 1 using ICP."""
    print("\n[STEP 2] SPATIAL ALIGNMENT (ICP)...")
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=pcd2,
        target=pcd1,
        max_correspondence_distance=2.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    print(f"  ✓ ICP Fitness: {reg_p2p.fitness:.6f}")
    print(f"  ✓ ICP RMSE: {reg_p2p.inlier_rmse:.6f}")
    
    pcd2_aligned = pcd2.transform(reg_p2p.transformation)
    
    return pcd1, pcd2_aligned

# ============= FEATURE EXTRACTION =============
def extract_geometric_features(pcd, points, radius=0.5):
    """
    Extract geometric features for each point using local neighborhood.
    Features:
    - Local point density
    - Height (z-coordinate)
    - Distance to nearest neighbor
    - Local planarity (eigen values)
    - Local roughness
    """
    print(f"\n[FEATURE EXTRACTION] Computing geometric features...")
    
    tree = KDTree(points)
    n_points = len(points)
    features = np.zeros((n_points, 8))
    
    # Compute normals for planarity features
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(pcd.normals)
    
    print(f"  Processing {n_points:,} points...")
    
    for i in range(n_points):
        if i % 10000 == 0:
            print(f"    Progress: {i:,}/{n_points:,} ({100*i/n_points:.1f}%)")
        
        point = points[i]
        
        # Feature 1: Height (z-coordinate)
        features[i, 0] = point[2]
        
        # Feature 2-3: X, Y coordinates
        features[i, 1] = point[0]
        features[i, 2] = point[1]
        
        # Find neighbors within radius
        indices = tree.query_ball_point(point, radius)
        
        # Feature 4: Local density
        features[i, 3] = len(indices) / (4/3 * np.pi * radius**3)
        
        # Feature 5: Distance to nearest neighbor
        if len(indices) > 1:
            neighbor_points = points[indices]
            distances = np.linalg.norm(neighbor_points - point, axis=1)
            distances = distances[distances > 0]  # Remove self
            features[i, 4] = np.min(distances) if len(distances) > 0 else 0
        else:
            features[i, 4] = radius
        
        # Feature 6: Mean distance to neighbors
        if len(indices) > 1:
            neighbor_points = points[indices]
            distances = np.linalg.norm(neighbor_points - point, axis=1)
            features[i, 5] = np.mean(distances)
        else:
            features[i, 5] = radius
        
        # Feature 7: Normal z-component (verticality)
        features[i, 6] = abs(normals[i, 2])
        
        # Feature 8: Local height variance
        if len(indices) > 1:
            neighbor_heights = points[indices][:, 2]
            features[i, 7] = np.var(neighbor_heights)
        else:
            features[i, 7] = 0
    
    print(f"  ✓ Extracted 8 geometric features per point")
    
    return features

# ============= ML-BASED CHANGE DETECTION =============
def ml_change_detection(pcd1, pcd2, feature_radius=0.5):
    """
    Machine Learning approach to change detection:
    1. Extract features from both scenes
    2. Use Isolation Forest for anomaly detection
    3. Use distance-based features for change scoring
    """
    print("\n[STEP 3] ML-BASED CHANGE DETECTION...")
    
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    
    # Extract features
    print("\n  [3.1] Extracting features from Scene 1...")
    features1 = extract_geometric_features(pcd1, pts1, radius=feature_radius)
    
    print("\n  [3.2] Extracting features from Scene 2...")
    features2 = extract_geometric_features(pcd2, pts2, radius=feature_radius)
    
    # Build KD-Trees for nearest neighbor queries
    print("\n  [3.3] Building spatial indices...")
    tree1 = KDTree(pts1)
    tree2 = KDTree(pts2)
    
    # ===== DETECT REMOVALS (Scene 1 → Scene 2) =====
    print("\n  [3.4] Detecting REMOVALS using ML...")
    
    # For each point in Scene 1, find distance to nearest point in Scene 2
    distances_1to2, indices_1to2 = tree2.query(pts1, k=1)
    
    # Create features for removal detection
    removal_features = np.column_stack([
        features1,
        distances_1to2.reshape(-1, 1),  # Distance to nearest in Scene 2
    ])
    
    # Normalize features
    scaler_removal = StandardScaler()
    removal_features_scaled = scaler_removal.fit_transform(removal_features)
    
    # Use Isolation Forest to detect anomalies (removed objects)
    iso_forest_removal = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_estimators=100
    )
    
    removal_predictions = iso_forest_removal.fit_predict(removal_features_scaled)
    removal_scores = iso_forest_removal.score_samples(removal_features_scaled)
    
    # Points with low anomaly scores + high distance = removals
    removal_candidates = np.where(
        (removal_predictions == -1) & (distances_1to2 > 0.6)  # Lowered threshold
    )[0]
    
    print(f"    Initial removal candidates: {len(removal_candidates):,}")
    
    # Cluster removal candidates
    if len(removal_candidates) > 0:
        removal_pcd = o3d.geometry.PointCloud()
        removal_pcd.points = o3d.utility.Vector3dVector(pts1[removal_candidates])
        labels = np.array(removal_pcd.cluster_dbscan(eps=0.4, min_points=30))  # More lenient
        
        removal_indices = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            if cluster_size >= 50:  # Lower minimum
                cluster_indices = removal_candidates[cluster_mask]
                removal_indices.extend(cluster_indices)
                print(f"    ✓ Removal cluster {label}: {cluster_size} points")
        
        removal_indices = np.array(removal_indices, dtype=int)
    else:
        removal_indices = np.array([], dtype=int)
    
    print(f"  ✓ Final removals: {len(removal_indices):,} points")
    
    # ===== DETECT ADDITIONS (Scene 2 → Scene 1) =====
    print("\n  [3.5] Detecting ADDITIONS using ML...")
    
    # For each point in Scene 2, find distance to nearest point in Scene 1
    distances_2to1, indices_2to1 = tree1.query(pts2, k=1)
    
    # Create features for addition detection
    addition_features = np.column_stack([
        features2,
        distances_2to1.reshape(-1, 1),  # Distance to nearest in Scene 1
    ])
    
    # Normalize features
    scaler_addition = StandardScaler()
    addition_features_scaled = scaler_addition.fit_transform(addition_features)
    
    # Use Isolation Forest to detect anomalies (added objects)
    iso_forest_addition = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_estimators=100
    )
    
    addition_predictions = iso_forest_addition.fit_predict(addition_features_scaled)
    addition_scores = iso_forest_addition.score_samples(addition_features_scaled)
    
    # Points with low anomaly scores + high distance = additions
    addition_candidates = np.where(
        (addition_predictions == -1) & (distances_2to1 > 0.6)  # Lowered threshold
    )[0]
    
    print(f"    Initial addition candidates: {len(addition_candidates):,}")
    
    # Cluster addition candidates
    if len(addition_candidates) > 0:
        addition_pcd = o3d.geometry.PointCloud()
        addition_pcd.points = o3d.utility.Vector3dVector(pts2[addition_candidates])
        labels = np.array(addition_pcd.cluster_dbscan(eps=0.4, min_points=30))  # More lenient
        
        addition_indices = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            if cluster_size >= 50:  # Lower minimum
                cluster_indices = addition_candidates[cluster_mask]
                addition_indices.extend(cluster_indices)
                print(f"    ✓ Addition cluster {label}: {cluster_size} points")
        
        addition_indices = np.array(addition_indices, dtype=int)
    else:
        addition_indices = np.array([], dtype=int)
    
    print(f"  ✓ Final additions: {len(addition_indices):,} points")
    
    return removal_indices, addition_indices

# ============= CREATE VISUALIZATION =============
def create_change_visualization(pcd1, pcd2, removal_indices, addition_indices):
    """
    Create change detection heatmap:
    - Gray: No change
    - Blue: Removals
    - Red: Additions
    """
    print("\n[STEP 4] CREATING VISUALIZATION...")
    
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    
    # Combine both scenes
    combined_points = np.vstack([pts1, pts2])
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    # Initialize colors (gray)
    colors = np.full((len(combined_points), 3), [0.5, 0.5, 0.5])
    
    # Color removals BLUE
    if len(removal_indices) > 0:
        colors[removal_indices] = [0.0, 0.4, 1.0]
        print(f"  ✓ Colored {len(removal_indices):,} points BLUE (removals)")
    
    # Color additions RED
    if len(addition_indices) > 0:
        colors[len(pts1) + addition_indices] = [1.0, 0.0, 0.0]
        print(f"  ✓ Colored {len(addition_indices):,} points RED (additions)")
    
    combined_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return combined_pcd

# ============= SAVE RESULTS =============
def save_results(pcd1, pcd2, change_pcd, removal_count, addition_count):
    """Save results."""
    print("\n[STEP 5] SAVING RESULTS...")
    
    scene1_path = os.path.join(OUTPUT_DIR, "ml_scene1_reference.ply")
    o3d.io.write_point_cloud(scene1_path, pcd1)
    
    scene2_path = os.path.join(OUTPUT_DIR, "ml_scene2_query.ply")
    o3d.io.write_point_cloud(scene2_path, pcd2)
    
    change_path = os.path.join(OUTPUT_DIR, "ml_change_detection.ply")
    o3d.io.write_point_cloud(change_path, change_pcd)
    
    stats_path = os.path.join(OUTPUT_DIR, "ml_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("ML-BASED CHANGE DETECTION STATISTICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Method: Isolation Forest + Feature Engineering\n")
        f.write(f"Features: 8 geometric features per point\n\n")
        f.write(f"Removals (BLUE): {removal_count:,} points\n")
        f.write(f"Additions (RED): {addition_count:,} points\n")
    
    print(f"  ✓ Saved: {change_path}")
    
    return scene1_path, scene2_path, change_path

# ============= 3-PANEL VISUALIZATION =============
def visualize_three_panels(pcd1, pcd2, change_pcd):
    """Display three panels side by side with synchronized controls."""
    print("\n[STEP 6] CREATING 3-PANEL VISUALIZATION...")
    
    # Color scenes
    pcd1_display = o3d.geometry.PointCloud(pcd1)
    colors1 = np.tile([0.7, 0.7, 0.7], (len(pcd1.points), 1))
    pcd1_display.colors = o3d.utility.Vector3dVector(colors1)
    
    pcd2_display = o3d.geometry.PointCloud(pcd2)
    colors2 = np.tile([0.7, 0.7, 0.7], (len(pcd2.points), 1))
    pcd2_display.colors = o3d.utility.Vector3dVector(colors2)
    
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="ML: Scene 1 (Reference)", width=853, height=720, left=0, top=100)
    vis1.add_geometry(pcd1_display)
    opt1 = vis1.get_render_option()
    opt1.point_size = 1.5
    opt1.background_color = np.array([0.1, 0.1, 0.1])
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="ML: Scene 2 (Query)", width=853, height=720, left=853, top=100)
    vis2.add_geometry(pcd2_display)
    opt2 = vis2.get_render_option()
    opt2.point_size = 1.5
    opt2.background_color = np.array([0.1, 0.1, 0.1])
    
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name="ML: Change Detection", width=854, height=720, left=1706, top=100)
    vis3.add_geometry(change_pcd)
    opt3 = vis3.get_render_option()
    opt3.point_size = 1.5
    opt3.background_color = np.array([0.1, 0.1, 0.1])
    
    # Get view controls
    ctr1 = vis1.get_view_control()
    ctr2 = vis2.get_view_control()
    ctr3 = vis3.get_view_control()
    
    vis1.reset_view_point(True)
    vis2.reset_view_point(True)
    vis3.reset_view_point(True)
    
    print("\n  3-Panel ML visualization ready!")
    print("  Method: Isolation Forest + Geometric Features")
    
    while True:
        if not vis1.poll_events():
            break
        if not vis2.poll_events():
            break
        if not vis3.poll_events():
            break
            
        # Sync camera
        param = ctr1.convert_to_pinhole_camera_parameters()
        ctr2.convert_from_pinhole_camera_parameters(param)
        ctr3.convert_from_pinhole_camera_parameters(param)
        
        vis1.update_renderer()
        vis2.update_renderer()
        vis3.update_renderer()
    
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    start_time = time.time()
    
    print("\n" + "="*100)
    
    # Load scenes
    pcd1, pcd2 = load_point_clouds()
    
    # Downsample
    print("\n[DOWNSAMPLING...]")
    pcd1 = downsample_point_cloud(pcd1, VOXEL_SIZE)
    pcd2 = downsample_point_cloud(pcd2, VOXEL_SIZE)
    print(f"  Scene 1: {len(pcd1.points):,} points")
    print(f"  Scene 2: {len(pcd2.points):,} points")
    
    # Register
    pcd1_reg, pcd2_reg = register_point_clouds(pcd1, pcd2)
    
    # ML-based change detection
    removal_indices, addition_indices = ml_change_detection(
        pcd1_reg, pcd2_reg, 
        feature_radius=FEATURE_RADIUS
    )
    
    # Create visualization
    change_pcd = create_change_visualization(pcd1_reg, pcd2_reg, removal_indices, addition_indices)
    
    # Save results
    scene1_file, scene2_file, change_file = save_results(
        pcd1_reg, pcd2_reg, change_pcd,
        len(removal_indices), len(addition_indices)
    )
    
    # Display
    visualize_three_panels(pcd1_reg, pcd2_reg, change_pcd)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*100)
    print("✓ ML-BASED CHANGE DETECTION COMPLETE")
    print("="*100)
    print(f"\nResults:")
    print(f"  Output: {change_file}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"\nML Method:")
    print(f"  • Isolation Forest anomaly detection")
    print(f"  • 8 geometric features per point")
    print(f"  • Feature normalization (StandardScaler)")
    print(f"  • DBSCAN clustering for post-processing")
    print(f"\nDetection Results:")
    print(f"  Removals (BLUE): {len(removal_indices):,} points")
    print(f"  Additions (RED): {len(addition_indices):,} points")
    print("\n" + "="*100 + "\n")
