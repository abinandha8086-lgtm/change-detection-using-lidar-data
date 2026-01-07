import os
import cv2
import torch
import numpy as np
import open3d as o3d
import glob
import matplotlib.cm as cm

BASE_DIR = "/home/abinandha/3d_pc_change/3DCDNet"
IMG_DIR  = os.path.join(BASE_DIR, "street_data")
SAVE_DIR = os.path.join(BASE_DIR, "depth_captures")

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============= ULTIMATE GEOMETRY PARAMETERS (MEGA-SCALE + REALISTIC) =============

# -------- SEDAN CAR DIMENSIONS (ULTRA-ENLARGED) --------
CAR_LENGTH = 4.85 * 1.35  # 6.55m (bigger)
CAR_WIDTH = 1.85 * 1.35   # 2.50m
CAR_HEIGHT = 1.55 * 1.35  # 2.09m
CAR_ROOF_HEIGHT = 1.25 * 1.35  # 1.69m
CAR_BUMPER_HEIGHT = 0.55 * 1.35  # 0.74m

# -------- TRUCK DIMENSIONS (LARGE COMMERCIAL TRUCK) --------
TRUCK_LENGTH = 8.5 * 1.35  # 11.48m (large truck)
TRUCK_WIDTH = 2.5 * 1.35   # 3.38m
TRUCK_HEIGHT = 3.5 * 1.35  # 4.73m
TRUCK_CABIN_HEIGHT = 2.8 * 1.35  # 3.78m
TRUCK_CARGO_HEIGHT = 3.2 * 1.35  # 4.32m

# -------- PEDESTRIAN DIMENSIONS (ULTRA-ENLARGED) --------
PERSON_HEIGHT = 1.75 * 1.80  # 3.15m (MUCH taller - 80% larger!)
PERSON_SHOULDER_WIDTH = 0.50 * 1.80  # 0.90m
PERSON_CHEST_WIDTH = 0.35 * 1.80  # 0.63m
PERSON_DEPTH = 0.28 * 1.80  # 0.50m
PERSON_HEAD_RADIUS = 0.12 * 1.80  # 0.216m
PERSON_ARM_LENGTH = 0.75 * 1.80  # 1.35m
PERSON_LEG_LENGTH = 0.95 * 1.80  # 1.71m

# -------- TREE DIMENSIONS (ULTRA-ENLARGED) --------
TREE_TRUNK_RADIUS = 0.35 * 1.40  # 0.49m
TREE_CANOPY_RADIUS = 5.5 * 1.40  # 7.70m
TREE_CANOPY_HEIGHT = 10.0 * 1.40  # 14.0m
TREE_CANOPY_BASE = 2.5 * 1.40  # 3.5m
TREE_BRANCH_DENSITY = 250

# -------- BUILDING DIMENSIONS (ULTRA-ENLARGED WITH DETAILS) --------
BUILDING_HEIGHT_MIN = 12.0 * 1.50  # 18.0m
BUILDING_HEIGHT_MAX = 22.0 * 1.50  # 33.0m
BUILDING_WINDOW_WIDTH = 1.5  # Window width in meters
BUILDING_WINDOW_HEIGHT = 1.2  # Window height in meters
BUILDING_WINDOW_SPACING = 2.0  # Space between windows
BUILDING_DOOR_WIDTH = 1.2  # Door width
BUILDING_DOOR_HEIGHT = 2.2  # Door height

# -------- SCENE SCALE (ABSOLUTE MAXIMUM SCALE) --------
SCENE_LENGTH = 250.0  # 0-250m (massive street)
SCENE_WIDTH = 50.0    # -25 to +25m (wide boulevard)
LIDAR_RANGE_MAX = 200.0  # Extended range

SIDEWALK_WIDTH = 3.0  # Wide sidewalks

print("[INFO] Loading MiDaS depth estimation model...")
midas = torch.hub.load(
    "intel-isl/MiDaS", "DPT_Large", trust_repo=True
).to(device).eval()

transform = torch.hub.load(
    "intel-isl/MiDaS", "transforms", trust_repo=True
).dpt_transform

# ============= LARGE COMMERCIAL TRUCK MODEL =============
def generate_commercial_truck(truck_x, truck_y, truck_z, truck_yaw=0.0):
    """Generate large commercial truck (cargo/delivery truck)."""
    points = []
    heights = []
    
    # -------- CABIN (front section) --------
    for _ in range(4000):
        local_x = np.random.uniform(TRUCK_LENGTH/2 - 2.8, TRUCK_LENGTH/2)
        local_y = np.random.uniform(-TRUCK_WIDTH/2, TRUCK_WIDTH/2)
        z_local = np.random.uniform(0.0, TRUCK_CABIN_HEIGHT)
        
        cos_y = np.cos(truck_yaw)
        sin_y = np.sin(truck_yaw)
        x = truck_x + local_x * cos_y - local_y * sin_y
        y = truck_y + local_x * sin_y + local_y * cos_y
        z = truck_z + z_local
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- CARGO BOX (rear section - large rectangular box) --------
    for _ in range(8000):
        local_x = np.random.uniform(-TRUCK_LENGTH/2, TRUCK_LENGTH/2 - 2.8)
        local_y = np.random.uniform(-TRUCK_WIDTH/2, TRUCK_WIDTH/2)
        z_local = np.random.uniform(0.0, TRUCK_CARGO_HEIGHT)
        
        cos_y = np.cos(truck_yaw)
        sin_y = np.sin(truck_yaw)
        x = truck_x + local_x * cos_y - local_y * sin_y
        y = truck_y + local_x * sin_y + local_y * cos_y
        z = truck_z + z_local
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- WINDSHIELD --------
    for _ in range(600):
        local_x = np.random.uniform(TRUCK_LENGTH/2 - 2.2, TRUCK_LENGTH/2 - 0.3)
        local_y = np.random.uniform(-TRUCK_WIDTH/2 + 0.2, TRUCK_WIDTH/2 - 0.2)
        z_local = TRUCK_CABIN_HEIGHT - 0.8 + (local_x / 3.0) * 0.15
        
        cos_y = np.cos(truck_yaw)
        sin_y = np.sin(truck_yaw)
        x = truck_x + local_x * cos_y - local_y * sin_y
        y = truck_y + local_x * sin_y + local_y * cos_y
        z = truck_z + z_local
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- WHEELS (6 wheels - truck) --------
    wheel_positions = [
        # Front wheels
        (TRUCK_LENGTH/2 - 1.5, -TRUCK_WIDTH/2 - 0.4),
        (TRUCK_LENGTH/2 - 1.5, TRUCK_WIDTH/2 + 0.4),
        # Middle rear wheels
        (-TRUCK_LENGTH/2 + 2.5, -TRUCK_WIDTH/2 - 0.4),
        (-TRUCK_LENGTH/2 + 2.5, TRUCK_WIDTH/2 + 0.4),
        # Back rear wheels
        (-TRUCK_LENGTH/2 + 1.0, -TRUCK_WIDTH/2 - 0.4),
        (-TRUCK_LENGTH/2 + 1.0, TRUCK_WIDTH/2 + 0.4),
    ]
    
    for wx, wy in wheel_positions:
        wheel_radius = 0.65
        wheel_width = 0.35
        
        for _ in range(500):
            theta = np.random.uniform(0, 2*np.pi)
            r = wheel_radius * (0.8 + 0.2*np.random.rand())
            z_offset = np.random.uniform(-wheel_width/2, wheel_width/2)
            
            local_x = wx + r * np.cos(theta)
            local_y = wy + z_offset
            z_local = wheel_radius
            
            cos_y = np.cos(truck_yaw)
            sin_y = np.sin(truck_yaw)
            x = truck_x + local_x * cos_y - local_y * sin_y
            y = truck_y + local_x * sin_y + local_y * cos_y
            z = truck_z + z_local
            
            points.append([x, y, z])
            heights.append(z)
    
    # -------- SIDE MIRRORS --------
    mirror_positions = [
        (TRUCK_LENGTH/2 - 1.5, -TRUCK_WIDTH/2 - 0.35),
        (TRUCK_LENGTH/2 - 1.5, TRUCK_WIDTH/2 + 0.35),
    ]
    
    for mx, my in mirror_positions:
        for _ in range(200):
            theta = np.random.uniform(0, 2*np.pi)
            r = 0.15 * (0.7 + 0.3*np.random.rand())
            
            local_x = mx + r * np.cos(theta)
            local_y = my + r * np.sin(theta)
            z_local = TRUCK_CABIN_HEIGHT - 0.5
            
            cos_y = np.cos(truck_yaw)
            sin_y = np.sin(truck_yaw)
            x = truck_x + local_x * cos_y - local_y * sin_y
            y = truck_y + local_x * sin_y + local_y * cos_y
            z = truck_z + z_local
            
            points.append([x, y, z])
            heights.append(z)
    
    return points, heights

# ============= ULTRA-HIGH-DETAIL CAR MODEL =============
def generate_sedan_car(car_x, car_y, car_z, car_yaw=0.0):
    """Generate ultra-detailed sedan car (MEGA-ENLARGED)."""
    points = []
    heights = []
    
    # -------- MAIN BODY (ultra-enlarged) --------
    for _ in range(6000):
        local_x = np.random.uniform(-CAR_LENGTH/2, CAR_LENGTH/2)
        local_y = np.random.uniform(-CAR_WIDTH/2, CAR_WIDTH/2)
        
        taper_factor = abs(local_x) / (CAR_LENGTH/2)
        if taper_factor > 0.85:
            if np.random.rand() > 0.3:
                continue
        
        if abs(local_x) < CAR_LENGTH/2 - 0.8:
            z_local = CAR_ROOF_HEIGHT + np.random.uniform(-0.30, 0.20)
        else:
            z_local = CAR_BUMPER_HEIGHT + np.random.uniform(-0.12, 0.40)
        
        cos_y = np.cos(car_yaw)
        sin_y = np.sin(car_yaw)
        x = car_x + local_x * cos_y - local_y * sin_y
        y = car_y + local_x * sin_y + local_y * cos_y
        z = car_z + z_local
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- WINDSHIELD (ultra-enlarged) --------
    for _ in range(1000):
        local_x = np.random.uniform(CAR_LENGTH/2 - 1.6, CAR_LENGTH/2 - 0.25)
        local_y = np.random.uniform(-CAR_WIDTH/2 + 0.15, CAR_WIDTH/2 - 0.15)
        z_local = CAR_ROOF_HEIGHT - 0.50 + (local_x / 4.0) * 0.12
        
        cos_y = np.cos(car_yaw)
        sin_y = np.sin(car_yaw)
        x = car_x + local_x * cos_y - local_y * sin_y
        y = car_y + local_x * sin_y + local_y * cos_y
        z = car_z + z_local
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- ROOF RACK (enhanced) --------
    for _ in range(400):
        local_x = np.random.uniform(-CAR_LENGTH/2 + 0.8, CAR_LENGTH/2 - 0.8)
        local_y = np.random.uniform(-CAR_WIDTH/2 + 0.20, CAR_WIDTH/2 - 0.20)
        z_local = CAR_ROOF_HEIGHT + 0.25 + np.random.rand() * 0.10
        
        cos_y = np.cos(car_yaw)
        sin_y = np.sin(car_yaw)
        x = car_x + local_x * cos_y - local_y * sin_y
        y = car_y + local_x * sin_y + local_y * cos_y
        z = car_z + z_local
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- SIDE MIRRORS (enlarged) --------
    mirror_positions = [
        (CAR_LENGTH/2 - 1.1, -CAR_WIDTH/2 - 0.25),
        (CAR_LENGTH/2 - 1.1, CAR_WIDTH/2 + 0.25),
    ]
    
    for mx, my in mirror_positions:
        mirror_radius = 0.12
        for _ in range(250):
            theta = np.random.uniform(0, 2*np.pi)
            r = mirror_radius * (0.7 + 0.3*np.random.rand())
            
            local_x = mx + r * np.cos(theta)
            local_y = my + r * np.sin(theta)
            z_local = CAR_ROOF_HEIGHT - 0.30
            
            cos_y = np.cos(car_yaw)
            sin_y = np.sin(car_yaw)
            x = car_x + local_x * cos_y - local_y * sin_y
            y = car_y + local_x * sin_y + local_y * cos_y
            z = car_z + z_local
            
            points.append([x, y, z])
            heights.append(z)
    
    # -------- WHEELS (mega-enlarged) --------
    wheel_positions = [
        (-CAR_LENGTH/2 + 1.2, -CAR_WIDTH/2 - 0.35),
        (-CAR_LENGTH/2 + 1.2, CAR_WIDTH/2 + 0.35),
        (CAR_LENGTH/2 - 1.2, -CAR_WIDTH/2 - 0.35),
        (CAR_LENGTH/2 - 1.2, CAR_WIDTH/2 + 0.35),
    ]
    
    for wx, wy in wheel_positions:
        wheel_radius = 0.52
        wheel_width = 0.32
        
        for _ in range(700):
            theta = np.random.uniform(0, 2*np.pi)
            r = wheel_radius * (0.8 + 0.2*np.random.rand())
            z_offset = np.random.uniform(-wheel_width/2, wheel_width/2)
            
            local_x = wx + r * np.cos(theta)
            local_y = wy + z_offset
            z_local = wheel_radius
            
            cos_y = np.cos(car_yaw)
            sin_y = np.sin(car_yaw)
            x = car_x + local_x * cos_y - local_y * sin_y
            y = car_y + local_x * sin_y + local_y * cos_y
            z = car_z + z_local
            
            points.append([x, y, z])
            heights.append(z)
    
    # -------- BUMPERS & GRILLE (enlarged) --------
    for _ in range(500):
        local_x = np.random.uniform(CAR_LENGTH/2 - 0.6, CAR_LENGTH/2)
        local_y = np.random.uniform(-CAR_WIDTH/2, CAR_WIDTH/2)
        z_local = CAR_BUMPER_HEIGHT + np.random.uniform(-0.15, 0.40)
        
        cos_y = np.cos(car_yaw)
        sin_y = np.sin(car_yaw)
        x = car_x + local_x * cos_y - local_y * sin_y
        y = car_y + local_x * sin_y + local_y * cos_y
        z = car_z + z_local
        
        points.append([x, y, z])
        heights.append(z)
    
    return points, heights

# ============= ULTRA-DETAILED PEDESTRIAN MODEL (MEGA-ENLARGED) =============
def generate_detailed_pedestrian(person_x, person_y, person_z, pose="standing"):
    """Generate MASSIVE, highly visible human with clear anatomical shape (80% larger)."""
    points = []
    heights = []
    
    # -------- HEAD (massive sphere - very dense) --------
    for _ in range(1200):
        phi = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, 2*np.pi)
        r = PERSON_HEAD_RADIUS * (0.75 + 0.25*np.random.rand())
        
        x = person_x + r * np.sin(phi) * np.cos(theta)
        y = person_y + r * np.sin(phi) * np.sin(theta)
        z = person_z + PERSON_HEIGHT - 0.35 + r * np.cos(phi)
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- NECK (thicker, more visible) --------
    for _ in range(350):
        theta = np.random.uniform(0, 2*np.pi)
        h = np.random.uniform(-0.35, 0.0)
        r_neck = 0.14  # Thicker neck
        
        x = person_x + r_neck * np.cos(theta)
        y = person_y + r_neck * np.sin(theta)
        z = person_z + PERSON_HEIGHT - 0.35 + h
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- SHOULDERS/UPPER TORSO (massive, very dense) --------
    for _ in range(2000):
        theta = np.random.uniform(0, 2*np.pi)
        h = np.random.uniform(0.40, 2.10)
        
        # Anatomically correct taper from shoulders to waist
        if h > 1.70:  # Upper shoulders - widest
            r_torso = PERSON_SHOULDER_WIDTH/2 * (0.90 + 0.10*np.random.rand())
        elif h > 1.20:  # Mid chest
            r_torso = PERSON_SHOULDER_WIDTH/2.2 * (0.85 + 0.15*np.random.rand())
        elif h > 0.70:  # Lower chest/upper abdomen
            r_torso = PERSON_SHOULDER_WIDTH/2.5 * (0.80 + 0.20*np.random.rand())
        else:  # Waist - narrowest
            r_torso = PERSON_SHOULDER_WIDTH/3.0 * (0.75 + 0.25*np.random.rand())
        
        x = person_x + r_torso * np.cos(theta)
        y = person_y + r_torso * np.sin(theta)
        z = person_z + h
        
        points.append([x, y, z])
        heights.append(z)
    
    # -------- CHEST/FRONT (anterior detail - anatomical shape) --------
    for _ in range(800):
        theta = np.random.uniform(-np.pi/3, np.pi/3)
        h = np.random.uniform(0.75, 1.70)
        r_chest = PERSON_CHEST_WIDTH/2 * (0.85 + 0.15*np.random.rand())
        
        # Front protrusion for realistic chest shape
        x = person_x + r_chest * np.cos(theta)
        y = person_y + r_chest * np.sin(theta) * 0.6  # Flatter on sides
        z = person_z + h
        
        points.append([x, y, z])
        heights.append(z)
    
    if pose == "standing":
        # -------- ARMS (massive, clearly visible) --------
        for arm_side in [-1, 1]:
            # Upper arm (shoulder to elbow)
            for _ in range(800):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(1.30, 2.10)
                r_arm = 0.14  # Thick upper arms
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.18) + r_arm * np.cos(theta)
                y = person_y + r_arm * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Forearm (elbow to wrist)
            for _ in range(650):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(0.45, 1.30)
                r_forearm = 0.13  # Slightly thinner forearms
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.24) + r_forearm * np.cos(theta)
                y = person_y + r_forearm * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Hands (denser clusters at wrists)
            for _ in range(300):
                theta = np.random.uniform(0, 2*np.pi)
                r_hand = 0.10
                h = np.random.uniform(0.35, 0.55)
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.24) + r_hand * np.cos(theta)
                y = person_y + r_hand * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
    
    elif pose == "walking":
        # -------- ARMS (walking swing motion) --------
        for arm_idx, arm_side in enumerate([-1, 1]):
            angle_offset = arm_idx * np.pi
            
            # Upper arm
            for _ in range(800):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(0.90, 2.10)
                r_arm = 0.14
                swing = 0.30 * np.cos(angle_offset)  # Wider swing
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.18 + swing) + r_arm * np.cos(theta)
                y = person_y + r_arm * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Forearm
            for _ in range(650):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(0.35, 1.30)
                r_forearm = 0.13
                swing = 0.30 * np.cos(angle_offset + np.pi/2)
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.30 + swing) + r_forearm * np.cos(theta)
                y = person_y + r_forearm * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
    
    elif pose == "arms_up":
        # -------- ARMS RAISED (very dramatic, highly visible) --------
        for arm_side in [-1, 1]:
            # Upper arm (pointing upward)
            for _ in range(900):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(1.70, 2.85)
                r_arm = 0.14
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.30) + r_arm * np.cos(theta)
                y = person_y + r_arm * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Forearm (reaching high)
            for _ in range(750):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(1.85, 3.15)  # Reaches above head
                r_forearm = 0.13
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.40) + r_forearm * np.cos(theta)
                y = person_y + r_forearm * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Hands raised (very visible at top)
            for _ in range(350):
                theta = np.random.uniform(0, 2*np.pi)
                r_hand = 0.11
                h = np.random.uniform(3.00, 3.25)
                
                x = person_x + arm_side * (PERSON_SHOULDER_WIDTH/2 + 0.40) + r_hand * np.cos(theta)
                y = person_y + r_hand * np.sin(theta) * 0.85
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
    
    # -------- HIPS/PELVIS (massive, anatomically correct) --------
    for _ in range(1100):
        theta = np.random.uniform(0, 2*np.pi)
        h = np.random.uniform(-0.20, 0.45)
        r_hip = PERSON_SHOULDER_WIDTH/2.5 * (0.85 + 0.15*np.random.rand())
        
        x = person_x + r_hip * np.cos(theta)
        y = person_y + r_hip * np.sin(theta)
        z = person_z + h
        
        points.append([x, y, z])
        heights.append(z)
    
    if pose == "standing":
        # -------- LEGS (massive cylindrical legs) --------
        for leg_side in [-PERSON_DEPTH/2, PERSON_DEPTH/2]:
            # Thighs (upper leg)
            for _ in range(1000):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(-1.00, 0.45)
                r_thigh = 0.20  # Thick thighs
                
                x = person_x + r_thigh * np.cos(theta)
                y = person_y + leg_side + r_thigh * np.sin(theta) * 0.9
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Lower legs (calves)
            for _ in range(850):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(-PERSON_LEG_LENGTH, -1.00)
                r_calf = 0.17  # Slightly thinner calves
                
                x = person_x + r_calf * np.cos(theta)
                y = person_y + leg_side + r_calf * np.sin(theta) * 0.9
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Feet (dense at ground level)
            for _ in range(400):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(-PERSON_LEG_LENGTH - 0.15, -PERSON_LEG_LENGTH + 0.05)
                r_foot = 0.15
                
                x = person_x + r_foot * np.cos(theta)
                y = person_y + leg_side + r_foot * np.sin(theta)
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
    
    elif pose == "walking":
        # -------- LEGS (walking stride - asymmetric) --------
        for leg_idx, leg_side in enumerate([-PERSON_DEPTH/2, PERSON_DEPTH/2]):
            stride = (-1)**(leg_idx + 1) * 0.40  # Wider stride
            
            # Thighs
            for _ in range(1000):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(-1.00, 0.45)
                r_thigh = 0.20
                
                x = person_x + r_thigh * np.cos(theta)
                y = person_y + leg_side + stride + r_thigh * np.sin(theta) * 0.9
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
            
            # Lower legs
            for _ in range(850):
                theta = np.random.uniform(0, 2*np.pi)
                h = np.random.uniform(-PERSON_LEG_LENGTH, -1.00)
                r_calf = 0.17
                
                x = person_x + r_calf * np.cos(theta)
                y = person_y + leg_side + stride + r_calf * np.sin(theta) * 0.9
                z = person_z + h
                
                points.append([x, y, z])
                heights.append(z)
    
    return points, heights

# ============= MEGA-REALISTIC TREE MODEL (ULTRA-ENLARGED) =============
def generate_mega_realistic_tree(tree_x, tree_y, tree_z):
    """Generate ultra-realistic enlarged tree with detailed branching."""
    points = []
    heights = []
    
    # -------- MAIN TRUNK (tapered, mega-enlarged) --------
    trunk_segments = 25
    for seg in range(trunk_segments):
        h_start = (seg / trunk_segments) * (TREE_CANOPY_BASE + 1.5)
        h_end = ((seg + 1) / trunk_segments) * (TREE_CANOPY_BASE + 1.5)
        
        taper = 1.0 - (seg / trunk_segments) * 0.40
        r_seg = TREE_TRUNK_RADIUS * taper
        
        for _ in range(400):
            theta = np.random.uniform(0, 2*np.pi)
            h = np.random.uniform(h_start, h_end)
            r = r_seg * (0.7 + 0.3*np.random.rand())
            
            x = tree_x + r * np.cos(theta)
            y = tree_y + r * np.sin(theta)
            z = tree_z + h
            
            points.append([x, y, z])
            heights.append(z)
    
    # -------- PRIMARY BRANCHES (mega-enlarged) --------
    num_primary = 10
    for branch_idx in range(num_primary):
        branch_theta = (branch_idx / num_primary) * 2 * np.pi
        branch_start = TREE_CANOPY_BASE * 0.35 + np.random.uniform(0, TREE_CANOPY_BASE * 0.40)
        branch_angle = np.random.uniform(0.35, 0.70)
        branch_length = np.random.uniform(3.5, 6.0)
        
        for seg in range(12):
            seg_frac = seg / 12.0
            h_b = seg_frac * branch_length
            
            x_b = branch_length * np.sin(branch_angle) * np.cos(branch_theta) * seg_frac
            y_b = branch_length * np.sin(branch_angle) * np.sin(branch_theta) * seg_frac
            z_b = branch_start + h_b * np.cos(branch_angle)
            
            r_b = 0.18 * (1.0 - seg_frac * 0.80)
            
            for _ in range(250):
                theta_b = np.random.uniform(0, 2*np.pi)
                r = r_b * (0.6 + 0.4*np.random.rand())
                
                x = tree_x + x_b + r * np.cos(theta_b)
                y = tree_y + y_b + r * np.sin(theta_b)
                z = tree_z + z_b
                
                points.append([x, y, z])
                heights.append(z)
    
    # -------- SECONDARY BRANCHES (mega-enlarged, many) --------
    num_secondary = 35
    for branch_idx in range(num_secondary):
        branch_theta = np.random.uniform(0, 2*np.pi)
        branch_start = TREE_CANOPY_BASE + np.random.uniform(0, TREE_CANOPY_HEIGHT * 0.75)
        branch_angle = np.random.uniform(0.25, 0.80)
        branch_length = np.random.uniform(2.0, 4.2)
        
        for seg in range(10):
            seg_frac = seg / 10.0
            h_b = seg_frac * branch_length
            
            x_b = branch_length * np.sin(branch_angle) * np.cos(branch_theta) * seg_frac
            y_b = branch_length * np.sin(branch_angle) * np.sin(branch_theta) * seg_frac
            z_b = branch_start + h_b * np.cos(branch_angle)
            
            r_b = 0.12 * (1.0 - seg_frac * 0.80)
            
            for _ in range(200):
                theta_b = np.random.uniform(0, 2*np.pi)
                r = r_b * (0.5 + 0.5*np.random.rand())
                
                x = tree_x + x_b + r * np.cos(theta_b)
                y = tree_y + y_b + r * np.sin(theta_b)
                z = tree_z + z_b
                
                points.append([x, y, z])
                heights.append(z)
    
    # -------- FOLIAGE CANOPY (mega-dense) --------
    canopy_center_z = tree_z + TREE_CANOPY_BASE + TREE_CANOPY_HEIGHT * 0.62
    
    for _ in range(12000):
        phi = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, 2*np.pi)
        
        if np.random.rand() > 0.36:
            continue
        
        r = TREE_CANOPY_RADIUS * (0.5 + 0.5*np.random.rand())
        
        x = tree_x + r * np.sin(phi) * np.cos(theta)
        y = tree_y + r * np.sin(phi) * np.sin(theta)
        z = canopy_center_z + r * np.cos(phi) * 0.85
        
        points.append([x, y, z])
        heights.append(z)
    
    return points, heights

# ============= REALISTIC BUILDING WITH WINDOWS AND DOORS =============
def generate_realistic_building_with_details(building_x, building_y, building_width, building_depth):
    """Generate realistic building facade with detailed windows and doors."""
    points = []
    heights = []
    
    height = np.random.uniform(BUILDING_HEIGHT_MIN, BUILDING_HEIGHT_MAX)
    
    # -------- MAIN FACADE (walls - dense) --------
    print(f"    [Building facade at ({building_x:.1f}, {building_y:.1f}), height: {height:.1f}m]")
    for _ in range(6000):
        x = np.random.uniform(building_x, building_x + building_width)
        y = np.random.uniform(building_y, building_y + building_depth)
        z = np.random.uniform(-1.5, height)
        
        # Dense walls (90% density)
        if np.random.rand() > 0.10:
            points.append([x, y, z])
            heights.append(z)
    
    # -------- WINDOWS (realistic grid pattern) --------
    window_rows = int(height / (BUILDING_WINDOW_HEIGHT + BUILDING_WINDOW_SPACING))
    window_cols = int(building_width / (BUILDING_WINDOW_WIDTH + BUILDING_WINDOW_SPACING))
    
    for row in range(window_rows):
        for col in range(window_cols):
            window_x = building_x + col * (BUILDING_WINDOW_WIDTH + BUILDING_WINDOW_SPACING) + 0.5
            window_y_start = building_y
            window_y_end = building_y + building_depth
            window_z = 1.5 + row * (BUILDING_WINDOW_HEIGHT + BUILDING_WINDOW_SPACING)
            
            # Sparse window points (glass reflection)
            if window_x < building_x + building_width - 0.5:
                for _ in range(150):
                    x = window_x + np.random.uniform(-BUILDING_WINDOW_WIDTH/2, BUILDING_WINDOW_WIDTH/2)
                    y = np.random.choice([window_y_start, window_y_end])
                    z = window_z + np.random.uniform(-BUILDING_WINDOW_HEIGHT/2, BUILDING_WINDOW_HEIGHT/2)
                    
                    if z < height:
                        points.append([x, y, z])
                        heights.append(z)
    
    # -------- DOORS (at ground level) --------
    door_cols = max(1, int(building_width / (BUILDING_DOOR_WIDTH * 3)))
    for door_idx in range(door_cols):
        door_x = building_x + (door_idx + 1) * (building_width / (door_cols + 1))
        door_y_center = building_y + building_depth / 2
        
        # Door frame (denser points)
        for _ in range(300):
            x = door_x + np.random.uniform(-BUILDING_DOOR_WIDTH/2, BUILDING_DOOR_WIDTH/2)
            y = door_y_center + np.random.uniform(-building_depth/2, building_depth/2)
            z = np.random.uniform(-1.5, BUILDING_DOOR_HEIGHT)
            
            points.append([x, y, z])
            heights.append(z)
    
    # -------- BUILDING EDGES (very dense corners for geometry) --------
    edge_points = 60
    for corner_x in [building_x, building_x + building_width]:
        for corner_y in [building_y, building_y + building_depth]:
            for z in np.linspace(-1.5, height, edge_points):
                for _ in range(8):
                    x = corner_x + np.random.randn() * 0.05
                    y = corner_y + np.random.randn() * 0.05
                    points.append([x, y, z])
                    heights.append(z)
    
    # -------- ROOF/TOP EDGE (very dense) --------
    for _ in range(1200):
        x = np.random.uniform(building_x, building_x + building_width)
        y = np.random.uniform(building_y, building_y + building_depth)
        z = height + np.random.uniform(-0.10, 0.15)
        
        points.append([x, y, z])
        heights.append(z)
    
    return points, heights

# ============= SCENE 1: WITH TRUCK + CAR + 3 PEOPLE =============
def generate_street_scene_1():
    """Scene 1: TRUCK + CAR + 3 PEOPLE"""
    all_points = []
    all_heights = []
    
    print("\n  [Generating ground plane - 250m × 50m...]")
    # -------- GROUND PLANE (MASSIVE, ULTRA-DENSE) --------
    ground_x = np.linspace(0, SCENE_LENGTH, 600)
    ground_y = np.linspace(-SCENE_WIDTH/2, SCENE_WIDTH/2, 300)
    
    for x in ground_x:
        for y in ground_y:
            z = -2.3 + np.random.randn() * 0.15
            all_points.append([x, y, z])
            all_heights.append(z)
    
    print("  [Generating ultra-wide sidewalks...]")
    # -------- SIDEWALKS (ULTRA-WIDE, ULTRA-DENSE) --------
    sidewalk_x = np.linspace(5, SCENE_LENGTH - 15, 500)
    
    for x in sidewalk_x:
        # Left sidewalk
        for y_offset in np.linspace(-SCENE_WIDTH/2, -SCENE_WIDTH/2 + SIDEWALK_WIDTH, 80):
            z = -2.1 + np.random.randn() * 0.08
            all_points.append([x, y_offset, z])
            all_heights.append(z)
        
        # Right sidewalk
        for y_offset in np.linspace(SCENE_WIDTH/2 - SIDEWALK_WIDTH, SCENE_WIDTH/2, 80):
            z = -2.1 + np.random.randn() * 0.08
            all_points.append([x, y_offset, z])
            all_heights.append(z)
    
    print("  [Generating detailed buildings with windows/doors...]")
    # -------- LEFT BUILDINGS (multiple) --------
    for building_idx in range(3):
        b_x = 50 + building_idx * 60
        building_pts, building_heights = generate_realistic_building_with_details(
            b_x, -SCENE_WIDTH/2 - 7, 55, 6
        )
        all_points.extend(building_pts)
        all_heights.extend(building_heights)
    
    # -------- RIGHT BUILDINGS (multiple) --------
    for building_idx in range(3):
        b_x = 50 + building_idx * 60
        building_pts, building_heights = generate_realistic_building_with_details(
            b_x, SCENE_WIDTH/2 + 1, 55, 6
        )
        all_points.extend(building_pts)
        all_heights.extend(building_heights)
    
    print("  [Generating 18 mega-trees...]")
    # -------- TREES (18 MEGA TREES) --------
    tree_positions = [
        [22, -SCENE_WIDTH/2 + 1.6, -2.3], [45, -SCENE_WIDTH/2 + 1.8, -2.3],
        [68, -SCENE_WIDTH/2 + 1.5, -2.3], [91, -SCENE_WIDTH/2 + 1.7, -2.3],
        [114, -SCENE_WIDTH/2 + 1.6, -2.3], [137, -SCENE_WIDTH/2 + 1.8, -2.3],
        [160, -SCENE_WIDTH/2 + 1.5, -2.3], [183, -SCENE_WIDTH/2 + 1.7, -2.3],
        [206, -SCENE_WIDTH/2 + 1.6, -2.3],
        [30, SCENE_WIDTH/2 - 1.7, -2.3], [53, SCENE_WIDTH/2 - 1.5, -2.3],
        [76, SCENE_WIDTH/2 - 1.8, -2.3], [99, SCENE_WIDTH/2 - 1.6, -2.3],
        [122, SCENE_WIDTH/2 - 1.7, -2.3], [145, SCENE_WIDTH/2 - 1.5, -2.3],
        [168, SCENE_WIDTH/2 - 1.8, -2.3], [191, SCENE_WIDTH/2 - 1.6, -2.3],
        [214, SCENE_WIDTH/2 - 1.7, -2.3]
    ]
    
    for tree_pos in tree_positions:
        tree_pts, tree_heights = generate_mega_realistic_tree(tree_pos[0], tree_pos[1], tree_pos[2])
        all_points.extend(tree_pts)
        all_heights.extend(tree_heights)
    
    print("  [Generating vehicles...]")
    # -------- LARGE TRUCK (100m ahead, center-left lane) - REMOVAL TARGET --------
    truck_pts, truck_heights = generate_commercial_truck(100, -3.0, -1.8, truck_yaw=0.05)
    all_points.extend(truck_pts)
    all_heights.extend(truck_heights)
    
    # -------- SEDAN CAR (150m ahead, center-right lane) --------
    car_pts, car_heights = generate_sedan_car(150, 3.5, -1.8, car_yaw=-0.08)
    all_points.extend(car_pts)
    all_heights.extend(car_heights)
    
    print("  [Generating pedestrians...]")
    # -------- PERSON 1 (40m ahead, left sidewalk, standing) --------
    person_pts, person_heights = generate_detailed_pedestrian(40, -SCENE_WIDTH/2 + 1.4, -1.8, pose="standing")
    all_points.extend(person_pts)
    all_heights.extend(person_heights)
    
    # -------- PERSON 2 (80m ahead, right sidewalk, walking) --------
    person_pts, person_heights = generate_detailed_pedestrian(80, SCENE_WIDTH/2 - 1.6, -1.8, pose="walking")
    all_points.extend(person_pts)
    all_heights.extend(person_heights)
    
    # -------- PERSON 3 (180m ahead, right sidewalk, standing) - REMOVAL TARGET --------
    person_pts, person_heights = generate_detailed_pedestrian(180, SCENE_WIDTH/2 - 1.5, -1.8, pose="standing")
    all_points.extend(person_pts)
    all_heights.extend(person_heights)
    
    print("  [Converting to point cloud and coloring...]")
    # Convert and color
    points = np.array(all_points)
    heights = np.array(all_heights)
    
    norm_h = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
    colors = cm.viridis(norm_h)[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# ============= SCENE 2: NO TRUCK + CAR + 4 DIFFERENT PEOPLE =============
def generate_street_scene_2():
    """Scene 2: NO TRUCK, but +2 NEW PEOPLE"""
    all_points = []
    all_heights = []
    
    print("\n  [Generating ground plane - 250m × 50m...]")
    # -------- GROUND PLANE --------
    ground_x = np.linspace(0, SCENE_LENGTH, 600)
    ground_y = np.linspace(-SCENE_WIDTH/2, SCENE_WIDTH/2, 300)
    
    for x in ground_x:
        for y in ground_y:
            z = -2.3 + np.random.randn() * 0.15
            all_points.append([x, y, z])
            all_heights.append(z)
    
    print("  [Generating ultra-wide sidewalks...]")
    # -------- SIDEWALKS --------
    sidewalk_x = np.linspace(5, SCENE_LENGTH - 15, 500)
    
    for x in sidewalk_x:
        for y_offset in np.linspace(-SCENE_WIDTH/2, -SCENE_WIDTH/2 + SIDEWALK_WIDTH, 80):
            z = -2.1 + np.random.randn() * 0.08
            all_points.append([x, y_offset, z])
            all_heights.append(z)
        
        for y_offset in np.linspace(SCENE_WIDTH/2 - SIDEWALK_WIDTH, SCENE_WIDTH/2, 80):
            z = -2.1 + np.random.randn() * 0.08
            all_points.append([x, y_offset, z])
            all_heights.append(z)
    
    print("  [Generating detailed buildings with windows/doors...]")
    # -------- BUILDINGS (multiple) --------
    for building_idx in range(3):
        b_x = 50 + building_idx * 60
        building_pts, building_heights = generate_realistic_building_with_details(
            b_x, -SCENE_WIDTH/2 - 7, 55, 6
        )
        all_points.extend(building_pts)
        all_heights.extend(building_heights)
    
    for building_idx in range(3):
        b_x = 50 + building_idx * 60
        building_pts, building_heights = generate_realistic_building_with_details(
            b_x, SCENE_WIDTH/2 + 1, 55, 6
        )
        all_points.extend(building_pts)
        all_heights.extend(building_heights)
    
    print("  [Generating 18 mega-trees...]")
    # -------- TREES (SAME 18 TREES) --------
    tree_positions = [
        [22, -SCENE_WIDTH/2 + 1.6, -2.3], [45, -SCENE_WIDTH/2 + 1.8, -2.3],
        [68, -SCENE_WIDTH/2 + 1.5, -2.3], [91, -SCENE_WIDTH/2 + 1.7, -2.3],
        [114, -SCENE_WIDTH/2 + 1.6, -2.3], [137, -SCENE_WIDTH/2 + 1.8, -2.3],
        [160, -SCENE_WIDTH/2 + 1.5, -2.3], [183, -SCENE_WIDTH/2 + 1.7, -2.3],
        [206, -SCENE_WIDTH/2 + 1.6, -2.3],
        [30, SCENE_WIDTH/2 - 1.7, -2.3], [53, SCENE_WIDTH/2 - 1.5, -2.3],
        [76, SCENE_WIDTH/2 - 1.8, -2.3], [99, SCENE_WIDTH/2 - 1.6, -2.3],
        [122, SCENE_WIDTH/2 - 1.7, -2.3], [145, SCENE_WIDTH/2 - 1.5, -2.3],
        [168, SCENE_WIDTH/2 - 1.8, -2.3], [191, SCENE_WIDTH/2 - 1.6, -2.3],
        [214, SCENE_WIDTH/2 - 1.7, -2.3]
    ]
    
    for tree_pos in tree_positions:
        tree_pts, tree_heights = generate_mega_realistic_tree(tree_pos[0], tree_pos[1], tree_pos[2])
        all_points.extend(tree_pts)
        all_heights.extend(tree_heights)
    
    print("  [Generating vehicles...]")
    # -------- TRUCK REMOVED (NO TRUCK IN SCENE 2) --------
    
    # -------- SEDAN CAR (150m ahead, center-right lane) - SAME AS SCENE 1 --------
    car_pts, car_heights = generate_sedan_car(150, 3.5, -1.8, car_yaw=-0.08)
    all_points.extend(car_pts)
    all_heights.extend(car_heights)
    
    print("  [Generating pedestrians...]")
    # -------- PERSON 1 (40m, left, standing) - SAME AS SCENE 1 --------
    person_pts, person_heights = generate_detailed_pedestrian(40, -SCENE_WIDTH/2 + 1.4, -1.8, pose="standing")
    all_points.extend(person_pts)
    all_heights.extend(person_heights)
    
    # -------- PERSON 2 (80m, right, walking) - SAME AS SCENE 1 --------
    person_pts, person_heights = generate_detailed_pedestrian(80, SCENE_WIDTH/2 - 1.6, -1.8, pose="walking")
    all_points.extend(person_pts)
    all_heights.extend(person_heights)
    
    # -------- PERSON 3 REMOVED (was at 180m) --------
    
    # -------- NEW PERSON 4 (120m, left sidewalk, arms up) - ADDITION TARGET --------
    person_pts, person_heights = generate_detailed_pedestrian(120, -SCENE_WIDTH/2 + 1.5, -1.8, pose="arms_up")
    all_points.extend(person_pts)
    all_heights.extend(person_heights)
    
    # -------- NEW PERSON 5 (200m, right sidewalk, walking) - ADDITION TARGET --------
    person_pts, person_heights = generate_detailed_pedestrian(200, SCENE_WIDTH/2 - 1.7, -1.8, pose="walking")
    all_points.extend(person_pts)
    all_heights.extend(person_heights)
    
    print("  [Converting to point cloud and coloring...]")
    # Convert and color
    points = np.array(all_points)
    heights = np.array(all_heights)
    
    norm_h = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
    colors = cm.viridis(norm_h)[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# ============= VISUALIZATION =============
def visualize_pcd(pcd, window_name="Point Cloud", point_size=0.8):
    """Professional visualization with optimal viewing angle for pedestrian visibility."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=2560, height=1440)  # Ultra-HD
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0, 0, 0])
    
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()

# ============= MAIN =============
if __name__ == "__main__":
    print("\n" + "="*100)
    print(" "*15 + "BIDIRECTIONAL CHANGE DETECTION - LIDAR POINT CLOUD GENERATION")
    print(" "*20 + "Scene 1: Truck + Car + 3 People")
    print(" "*20 + "Scene 2: NO Truck + Car + 4 Different People")
    print("="*100)
    
    print(f"\n[BIDIRECTIONAL CHANGE TARGETS]")
    print(f"  REMOVALS (in Scene 1, NOT in Scene 2) → BLUE:")
    print(f"    • Large commercial truck at 100m, center-left")
    print(f"    • Person standing at 180m, right sidewalk")
    print(f"  ADDITIONS (in Scene 2, NOT in Scene 1) → RED:")
    print(f"    • MASSIVE person (3.15m tall!) with arms up at 120m, left sidewalk")
    print(f"    • MASSIVE person (3.15m tall!) walking at 200m, right sidewalk")
    print(f"  UNCHANGED:")
    print(f"    • Sedan car at 150m")
    print(f"    • Person at 40m (left)")
    print(f"    • Person at 80m (right)")
    print(f"\n  [HUMAN SIZE UPGRADE: 80% LARGER!]")
    print(f"    • Height: 3.15m (was 2.45m)")
    print(f"    • Shoulder width: 0.90m")
    print(f"    • Very detailed anatomy: head, neck, torso, arms, hands, hips, thighs, calves, feet")
    print(f"    • Point density: 8,000-10,000 points per person (ultra-high)")
    print(f"    • Arms-up pose reaches 3.25m height - EXTREMELY VISIBLE!")
    
    print("\n" + "-"*100)
    print("[1/2] SCENE 1: TRUCK + CAR + 3 PEOPLE")
    print("-"*100)
    
    print("\n  Generating Scene 1...")
    pcd_scene1 = generate_street_scene_1()
    scene1_path = os.path.join(SAVE_DIR, "ultimate_scene_1_250m_detailed.ply")
    o3d.io.write_point_cloud(scene1_path, pcd_scene1)
    
    print(f"\n[✓] Scene 1 COMPLETE")
    print(f"    File: ultimate_scene_1_250m_detailed.ply")
    print(f"    Total points: {len(pcd_scene1.points):,}")
    print(f"    Contents:")
    print(f"      • 1 large commercial truck (100m) → WILL BE REMOVED")
    print(f"      • 1 sedan car (150m) → STAYS")
    print(f"      • 3 pedestrians (40m, 80m, 180m) → 180m WILL BE REMOVED")
    
    print("\n" + "-"*100)
    print("[2/2] SCENE 2: NO TRUCK + CAR + 4 DIFFERENT PEOPLE")
    print("-"*100)
    
    print("\n  Generating Scene 2...")
    pcd_scene2 = generate_street_scene_2()
    scene2_path = os.path.join(SAVE_DIR, "ultimate_scene_2_250m_3people.ply")
    o3d.io.write_point_cloud(scene2_path, pcd_scene2)
    
    print(f"\n[✓] Scene 2 COMPLETE")
    print(f"    File: ultimate_scene_2_250m_3people.ply")
    print(f"    Total points: {len(pcd_scene2.points):,}")
    print(f"    Contents:")
    print(f"      • 0 trucks (REMOVED)")
    print(f"      • 1 sedan car (150m) → STAYS")
    print(f"      • 4 MASSIVE pedestrians (3.15m tall, 80% larger!):")
    print(f"          - 40m, 80m (SAME as Scene 1)")
    print(f"          - 120m (NEW - arms raised to 3.25m!) → ADDITION - ULTRA VISIBLE")
    print(f"          - 200m (NEW - walking) → ADDITION - ULTRA VISIBLE")
    print(f"      • Each new person: ~8,500 points with anatomical detail")
    
    print("\n" + "="*100)
    print("CHANGE DETECTION SUMMARY")
    print("="*100)
    
    print(f"\n  Expected detection results:")
    print(f"    🔵 BLUE (Removals): Truck (100m) + Person (180m)")
    print(f"    🔴 RED (Additions): Person (120m) + Person (200m)")
    print(f"    ⚪ Gray (No change): Car (150m) + Persons (40m, 80m) + all buildings/trees")
    
    print("\n" + "="*100)
    print("VISUALIZATION")
    print("="*100)
    
    print("\n[>>> Displaying Scene 1 <<<]")
    visualize_pcd(pcd_scene1, "Scene 1: Truck + Car + 3 People", point_size=0.7)
    
    print("\n[>>> Displaying Scene 2 <<<]")
    visualize_pcd(pcd_scene2, "Scene 2: No Truck + Car + 4 People", point_size=0.7)
    
    print("\n" + "="*100)
    print("[GENERATION COMPLETE]")
    print("="*100)
    print(f"\n  Files saved: {SAVE_DIR}")
    print(f"  Ready for 3D change detection!")
    print("\n" + "="*100 + "\n")
