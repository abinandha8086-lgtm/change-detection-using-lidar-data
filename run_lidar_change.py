import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# ============= PARAMETERS FOR FULL TRUCK RECOVERY =============
BASE_DIR = "/home/abinandha/3d_pc_change/3DCDNet"
SAVE_DIR = os.path.join(BASE_DIR, "depth_captures")

VOXEL_SIZE = 0.03          
CHANGE_THRESHOLD = 0.45    
CLUSTER_EPS = 0.7          
MIN_POINTS_FOR_OBJECT = 100

def recover_full_objects(source_pcd, seed_indices, stage_name):
    """
    Takes the small 'change' points and pulls the entire 
    original cluster they belong to with detailed logging.
    """
    if len(seed_indices) == 0:
        print(f"  [LOG] No seed points found for {stage_name}. Skipping recovery.")
        return o3d.geometry.PointCloud()

    print(f"  [LOG] Processing {stage_name}: Growing {len(seed_indices)} seeds into full shapes...")

    # 1. Cluster the entire original source cloud
    labels = np.array(source_pcd.cluster_dbscan(eps=CLUSTER_EPS, min_points=30))
    num_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    print(f"  [LOG] Found {num_clusters} total clusters in the source data.")
    
    # 2. Identify which clusters contain our 'seeds'
    seed_labels = labels[seed_indices]
    unique_labels = np.unique(seed_labels)
    unique_labels = unique_labels[unique_labels != -1] # Remove noise label
    
    print(f"  [LOG] Seeds overlap with {len(unique_labels)} specific clusters.")
    
    final_indices = []
    pts = np.asarray(source_pcd.points)
    recovered_count = 0

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        z_max = np.max(pts[cluster_indices, 2])
        
        # Validation: Size and Height
        if z_max <= 3.5 and len(cluster_indices) > MIN_POINTS_FOR_OBJECT:
            final_indices.extend(cluster_indices)
            recovered_count += 1
            print(f"    -> Kept Cluster {label}: {len(cluster_indices)} points, Max Height: {z_max:.2f}m")
        else:
            reason = "Too Tall" if z_max > 3.5 else "Too Small"
            print(f"    -> Dropped Cluster {label}: {reason} ({z_max:.2f}m, {len(cluster_indices)} pts)")
        
    print(f"  [LOG] Final {stage_name} result: Recovered {recovered_count} solid object(s).")
    return source_pcd.select_by_index(final_indices)

def detect_and_recover(pcd_ref, pcd_query):
    def get_seeds(src, tar, name):
        print(f"\n[STEP] Finding change seeds for {name}...")
        src_pts = np.asarray(src.points)
        tar_pts = np.asarray(tar.points)
        tree = KDTree(tar_pts)
        dist, _ = tree.query(src_pts, k=1)
        
        seeds = np.where(dist > CHANGE_THRESHOLD)[0]
        print(f"  [LOG] Initial distance check: {len(seeds)} points differ.")
        
        if len(seeds) < 20: 
            return []
        
        temp = src.select_by_index(seeds)
        _, clean_idx = temp.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        print(f"  [LOG] Statistical cleaning: {len(clean_idx)} seeds remaining.")
        return seeds[clean_idx]

    add_seeds = get_seeds(pcd_query, pcd_ref, "ADDITIONS (Red)")
    rem_seeds = get_seeds(pcd_ref, pcd_query, "REMOVALS (Blue)")
    
    full_addition = recover_full_objects(pcd_query, add_seeds, "ADDITIONS")
    full_removal = recover_full_objects(pcd_ref, rem_seeds, "REMOVALS")
    
    return full_addition, full_removal

if __name__ == "__main__":
    print("=== STARTING 3D CHANGE DETECTION ===")
    
    print("[1/4] Loading Point Clouds...")
    pcd1 = o3d.io.read_point_cloud(os.path.join(SAVE_DIR, "ultimate_scene_1_250m_detailed.ply")).voxel_down_sample(VOXEL_SIZE)
    pcd2 = o3d.io.read_point_cloud(os.path.join(SAVE_DIR, "ultimate_scene_2_250m_3people.ply")).voxel_down_sample(VOXEL_SIZE)
    
    print("[2/4] Aligning scenes via ICP...")
    reg = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, 0.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    pcd2.transform(reg.transformation)
    print("  [LOG] Alignment complete.")
    
    print("[3/4] Extracting Full Object Shapes...")
    add_full, rem_full = detect_and_recover(pcd1, pcd2)
    
    print("[4/4] Preparing Visualization...")
    pcd2.paint_uniform_color([0.2, 0.2, 0.2]) 
    add_full.paint_uniform_color([1.0, 0.0, 0.0]) 
    rem_full.paint_uniform_color([0.0, 0.5, 1.0]) 
    
    print("=== PROCESSING COMPLETE. OPENING VIEWER ===")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Pint cloud Change Detection", width=1200, height=800)
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().point_size = 2.0
    
    vis.add_geometry(pcd2)
    vis.add_geometry(add_full)
    vis.add_geometry(rem_full)
    vis.run()
    vis.destroy_window()
