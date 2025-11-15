import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import csv
from scipy.spatial.transform import Rotation as R

class ParcelDetector:
    def __init__(self, rgb_path, depth_path, ply_path, color_intrinsics, depth_intrinsics, extrinsics):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.ply_path = ply_path
        self.color_intrinsics = color_intrinsics
        self.depth_intrinsics = depth_intrinsics
        self.extrinsics = extrinsics
        self.roi = (560, 150, 300, 330)
        self.conveyor_plane = None
        
    def load_data(self):
        self.rgb = cv2.imread(str(self.rgb_path))
        self.depth = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
        self.pcd = o3d.io.read_point_cloud(str(self.ply_path))
        
    def extract_roi(self):
        x, y, w, h = self.roi
        self.rgb_roi = self.rgb[y:y+h, x:x+w]
        self.depth_roi = self.depth[y:y+h, x:x+w]
        
    def depth_to_camera_coords(self, u, v, depth_value):
        fx = self.color_intrinsics['fx']
        fy = self.color_intrinsics['fy']
        cx = self.color_intrinsics['cx']
        cy = self.color_intrinsics['cy']
        Z = depth_value / 1000.0
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        return np.array([X, Y, Z])
    
    def transform_to_color_frame(self, point_depth_cam):
        R_matrix = self.extrinsics['R']
        t_vector = self.extrinsics['t']
        point_color_cam = R_matrix @ point_depth_cam + t_vector
        return point_color_cam
    
    def pixels_to_3d_points(self, x_coords, y_coords, depth_values):
        points_3d = []
        for i in range(len(x_coords)):
            if depth_values[i] > 0:  
                point_depth = self.depth_to_camera_coords(x_coords[i], y_coords[i], depth_values[i])
                point_color = self.transform_to_color_frame(point_depth)
                points_3d.append(point_color)
        return np.array(points_3d)
    
    def fit_conveyor_plane_ransac(self, conveyor_mask):
        y_coords, x_coords = np.where(conveyor_mask == 255)
        x_full = x_coords + self.roi[0]
        y_full = y_coords + self.roi[1]
        depth_values = self.depth[y_full, x_full]
        conveyor_points_3d = self.pixels_to_3d_points(x_full, y_full, depth_values)
        
        if len(conveyor_points_3d) < 100:
            print("Warning: Not enough conveyor points for RANSAC, using default plane")
            avg_z = np.mean(conveyor_points_3d[:, 2]) if len(conveyor_points_3d) > 0 else 1.0
            return np.array([0, 0, 1, -avg_z])
        print(f"Fitting plane with {len(conveyor_points_3d)} conveyor points...")
        
        best_inliers = 0
        best_plane = None
        n_iterations = 100
        threshold = 0.005  
        for _ in range(n_iterations):
            idx = np.random.choice(len(conveyor_points_3d), 3, replace=False)
            pts = conveyor_points_3d[idx]
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) < 1e-6:
                continue
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, pts[0])
            distances = np.abs(conveyor_points_3d @ normal + d)
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d])
        print(f"RANSAC: {best_inliers}/{len(conveyor_points_3d)} inliers")
        print(f"Plane model: {best_plane}")
        return best_plane
    
    def rgb_list_to_hsv_ranges(self, rgb_list, dh=8, ds=60, dv=60):
        hsv_ranges = []
        for (r, g, b) in rgb_list:
            bgr = np.uint8([[[b, g, r]]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
            s_low  = max(0,   s - ds)
            s_high = min(255, s + ds)
            v_low  = max(0,   v - dv)
            v_high = min(255, v + dv)

            h_low  = (h - dh) % 180
            h_high = (h + dh) % 180

            if h_low <= h_high:
                lower = np.array([h_low,  s_low, v_low],  dtype=np.uint8)
                upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
                hsv_ranges.append((lower, upper))
            else:
                lower1 = np.array([0,     s_low, v_low],  dtype=np.uint8)
                upper1 = np.array([h_high, s_high, v_high], dtype=np.uint8)
                lower2 = np.array([h_low, s_low, v_low],  dtype=np.uint8)
                upper2 = np.array([179,   s_high, v_high], dtype=np.uint8)
                hsv_ranges.append((lower1, upper1))
                hsv_ranges.append((lower2, upper2))
        return hsv_ranges
    
    def mask_from_color_list(self, hsv_img, hsv_ranges):
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in hsv_ranges:
            mask |= cv2.inRange(hsv_img, lower, upper)
        return mask
    
    
    def segment_parcels(self):
        hsv = cv2.cvtColor(self.rgb_roi, cv2.COLOR_BGR2HSV)

        lower_green = np.array([45, 30, 15], dtype=np.uint8)
        upper_green = np.array([85, 255, 140], dtype=np.uint8)
        conveyor_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        mask2 = cv2.inRange(hsv, (50, 40, 20), (80, 200, 120))
        conveyor_mask = cv2.bitwise_or(conveyor_mask, mask2)
        extra_greens_rgb = [
            (0, 139, 0),
            (0, 100, 0),
            (0, 139, 69),
            (84, 139, 84),
            (46, 139, 87),
            (105, 139, 105),
            (34, 139, 34),
            (69, 139, 0),
            (85, 107, 47),
        ]
        hsv_ranges = self.rgb_list_to_hsv_ranges(extra_greens_rgb, dh=8, ds=60, dv=60)
        extra_mask = self.mask_from_color_list(hsv, hsv_ranges)
        conveyor_mask = cv2.bitwise_or(conveyor_mask, extra_mask)
        kernel = np.ones((5, 5), np.uint8)
        conveyor_mask = cv2.morphologyEx(conveyor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        print("\nFitting conveyor plane...")
        self.conveyor_plane = self.fit_conveyor_plane_ransac(conveyor_mask)

        parcel_mask = cv2.bitwise_not(conveyor_mask)
        parcel_mask = cv2.morphologyEx(parcel_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        parcel_mask = cv2.morphologyEx(parcel_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(parcel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        return contours
    
    def get_contour_3d_points(self, contour):
        mask = np.zeros(self.depth_roi.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)       
        y_roi, x_roi = np.where(mask == 255)
        
        if len(y_roi) == 0:
            return None
        x_color = x_roi + self.roi[0]
        y_color = y_roi + self.roi[1]
        
        depth_values = self.depth[y_color, x_color]
        points_3d = []
        for i in range(len(x_color)):
            if depth_values[i] > 0:
                point_depth = self.depth_to_camera_coords(
                    x_color[i], y_color[i], depth_values[i]
                )
                points_3d.append(point_depth)
        return np.array(points_3d) if len(points_3d) > 0 else None
    
    def calculate_height_from_plane(self, points_3d, plane_model):
        a, b, c, d = plane_model
        numerator = np.abs(points_3d @ np.array([a, b, c]) + d)
        denominator = np.sqrt(a**2 + b**2 + c**2)
        heights = numerator / denominator
        return heights
    
    def find_highest_parcel(self, contours):
        max_height = -np.inf
        highest_contour = None
        highest_points_3d = None
        print("\nAnalyzing parcel heights...")
        for i, cnt in enumerate(contours):
            points_3d = self.get_contour_3d_points(cnt)
            if points_3d is None or len(points_3d) < 10:
                continue
            heights = self.calculate_height_from_plane(points_3d, self.conveyor_plane)
            height_90 = np.percentile(heights, 90)
            avg_height = np.mean(heights)
            max_h = np.max(heights)
            print(f"Parcel {i+1}: avg={avg_height*1000:.1f}mm, p90={height_90*1000:.1f}mm, max={max_h*1000:.1f}mm")

            if height_90 > max_height:
                max_height = height_90
                highest_contour = cnt
                highest_points_3d = points_3d
        
        if highest_contour is not None:
            print(f"\n Highest parcel found with height: {max_height*1000:.1f}mm")
        
        return highest_contour, highest_points_3d, max_height
    
    def filter_top_surface_points(self, points_3d, height_threshold_ratio=0.8):
        heights = self.calculate_height_from_plane(points_3d, self.conveyor_plane)
        height_threshold = np.percentile(heights, height_threshold_ratio * 100)
        top_mask = heights >= height_threshold
        top_points = points_3d[top_mask]
        print(f"Filtered top surface: {len(top_points)}/{len(points_3d)} points ({100*len(top_points)/len(points_3d):.1f}%)")
        
        return top_points
    
    def remove_statistical_outliers(self, points_3d, nb_neighbors=20, std_ratio=2.0):
        if len(points_3d) < nb_neighbors:
            return points_3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        clean_points = np.asarray(cl.points)
        print(f"Outlier removal: {len(clean_points)}/{len(points_3d)} points kept")
        return clean_points
    
    def fit_top_surface_plane(self, points_3d):
        if len(points_3d) < 10:
            print("Warning: Too few points for plane fitting")
            return np.array([0, 0, 1]), np.array([0, 0, 1, 0])
        
        print(f"Fitting top surface plane with {len(points_3d)} points...")
        best_inliers = 0
        best_plane = None
        best_inlier_mask = None
        n_iterations = 100  
        threshold = 0.005  
        
        for iterations in range(n_iterations):
            idx = np.random.choice(len(points_3d), 3, replace=False)
            pts = points_3d[idx]
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
            
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, pts[0])
            
            if normal[2] < 0:
                normal = -normal
                d = -d
            
            distances = np.abs(points_3d @ normal + d)
            inlier_mask = distances < threshold
            inliers = np.sum(inlier_mask)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d])
                best_inlier_mask = inlier_mask
        
        if best_plane is None or best_inliers < 10:
            print("Warning: RANSAC failed, using default upward normal")
            return np.array([0, 0, 1]), np.array([0, 0, 1, 0])
        
        print(f"RANSAC result: {best_inliers}/{len(points_3d)} inliers ({100*best_inliers/len(points_3d):.1f}%)")
        inlier_points = points_3d[best_inlier_mask]
        
        if len(inlier_points) >= 10:
            print(f"Refining plane with {len(inlier_points)} inliers using SVD...")
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            U, S, Vt = np.linalg.svd(centered)
            normal_refined = Vt[-1]  
            if normal_refined[2] < 0:
                normal_refined = -normal_refined
            
            d_refined = -np.dot(normal_refined, centroid)
            best_plane = np.array([normal_refined[0], normal_refined[1], normal_refined[2], d_refined])
            print(f"Refined normal: ({normal_refined[0]:.4f}, {normal_refined[1]:.4f}, {normal_refined[2]:.4f})")
        normal_vector = best_plane[:3]
        print(f"Final top surface normal: ({normal_vector[0]:.4f}, {normal_vector[1]:.4f}, {normal_vector[2]:.4f})")
        return normal_vector, best_plane
    
    def normal_vector_to_euler(self, normal_vector):
        n = normal_vector / np.linalg.norm(normal_vector)
        z_axis = n
        if abs(z_axis[2]) < 0.9:
            x_temp = np.array([0, 0, 1])
        else:
            x_temp = np.array([1, 0, 0])
        y_axis = np.cross(z_axis, x_temp)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        try:
            rot = R.from_matrix(rotation_matrix)
            euler_angles = rot.as_euler('xyz', degrees=True)
        except:
            print("Warning: Failed to convert normal to Euler angles")
            Rx = np.arctan2(n[1], n[2]) * 180 / np.pi
            Ry = np.arctan2(-n[0], np.sqrt(n[1]**2 + n[2]**2)) * 180 / np.pi
            Rz = 0.0
            euler_angles = np.array([Rx, Ry, Rz])
        
        print(f"Euler angles from normal: Rx={euler_angles[0]:.2f}°, Ry={euler_angles[1]:.2f}°, Rz={euler_angles[2]:.2f}°")
        
        return euler_angles
    
    def get_3d_center_and_orientation(self, contour, points_3d_raw):
        points_3d = self.remove_statistical_outliers(points_3d_raw)
        if len(points_3d) < 10:
            print("Warning: Too few points after outlier removal")
            points_3d = points_3d_raw
        top_points = self.filter_top_surface_points(points_3d)
        if len(top_points) < 10:
            print("Warning: Too few top surface points, using all points")
            top_points = points_3d
        
        center_3d = np.mean(top_points, axis=0)
        
        print(f"Center 3D: ({center_3d[0]:.4f}, {center_3d[1]:.4f}, {center_3d[2]:.4f})")
        normal_vector, top_plane = self.fit_top_surface_plane(top_points)
        euler_angles = self.normal_vector_to_euler(normal_vector)
        print(f"Orientation: Rx={euler_angles[0]:.2f}°, Ry={euler_angles[1]:.2f}°, Rz={euler_angles[2]:.2f}°")
        return center_3d, euler_angles
    
    def visualize_result(self, contour, center_3d, rotation, height):
        result_img = self.rgb_roi.copy()
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 3)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(result_img, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(result_img, f"Height: {height*1000:.1f}mm", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_img, f"X: {center_3d[0]:.3f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, f"Y: {center_3d[1]:.3f}m", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, f"Z: {center_3d[2]:.3f}m", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        rect = cv2.minAreaRect(contour.astype(np.float32))
        (cx, cy), (w, h), ang = rect

        if np.isfinite([cx, cy, w, h]).all() and w > 0 and h > 0:
            box = cv2.boxPoints(rect)                     
            box = np.round(box).astype(np.int32)        
            if box.shape[0] >= 3:
                cv2.drawContours(result_img, [box], 0, (255, 0, 0), 2)
        else:
            x, y, bw, bh = cv2.boundingRect(contour.astype(np.int32))
            if bw > 0 and bh > 0:
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        
        return result_img
    
    def process(self):
        
        self.load_data()
        self.extract_roi()
        contours = self.segment_parcels()
        if len(contours) == 0:
            return None
        print(f"\n Found {len(contours)} parcel(s)")
        highest_contour, highest_points_3d, max_height = self.find_highest_parcel(contours)
        if highest_contour is None:
            return None
        center_3d, rotation = self.get_3d_center_and_orientation(highest_contour, highest_points_3d)
        result_img = self.visualize_result(highest_contour, center_3d, rotation, max_height)
        print(f"Center (x, y, z): ({center_3d[0]:.4f}, {center_3d[1]:.4f}, {center_3d[2]:.4f}) m")
        print(f"Rotation (Rx, Ry, Rz): ({rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}) deg")
        print(f"Height from conveyor: {max_height*1000:.1f} mm")
        
        return {
            'center': center_3d,
            'rotation': rotation,
            'height': max_height,
            'visualization': result_img
        }


def load_camera_params():
    color_intrinsics = {
        'width': 1280,
        'height': 720,
        'fx': 643.90087890625,
        'fy': 643.136535644531,
        'cx': 650.211303710937,
        'cy': 355.795593261718,
        'model': 'distortion.inverse_brown_conrady',
        'coeffs': [-0.0565845072269439, 0.0654425566296701, 
                   -0.000869411334861069, 0.000167517995578236, 
                   -0.0209577456116763]
    }
    
    depth_intrinsics = {
        'width': 1280,
        'height': 720,
        'fx': 650.061645507812,
        'fy': 650.061645507812,
        'cx': 649.592895507812,
        'cy': 360.941558378906,
        'model': 'distortion.brown_conrady',
        'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    R_matrix = np.array([
        [0.9999898076057434, -0.00020347206736914814, -0.004507721401751041],
        [0.00018898719281423837, 0.9999948143959045, -0.0032135415822267532],
        [0.004508351907134056, 0.003212657058611512, 0.9999846816062927]
    ])
    
    t_vector = np.array([
        -0.05905,
        8.67399e-5,
        0.00041
    ])
    
    extrinsics = {
        'R': R_matrix,
        't': t_vector
    }
    
    return color_intrinsics, depth_intrinsics, extrinsics


def process_dataset(rgb_dir, depth_dir, ply_dir, output_csv):
    color_intrinsics, depth_intrinsics, extrinsics = load_camera_params()
    
    rgb_path = Path(rgb_dir)
    depth_path = Path(depth_dir)
    ply_path = Path(ply_dir)
    
    results = []
    rgb_files = sorted(rgb_path.glob('*.png')) + sorted(rgb_path.glob('*.jpg'))
    
    for idx, rgb_file in enumerate(rgb_files):
        print(f" Processing {idx+1}/{len(rgb_files)}: {rgb_file.name}")
        
        base_name = rgb_file.stem
        depth_file = depth_path / f"{base_name}.png"
        ply_file = ply_path / f"{base_name}.ply"
        
        if not depth_file.exists() or not ply_file.exists():
            print(f"✗ Missing depth or ply file for {rgb_file.name}")
            continue
        detector = ParcelDetector(
            rgb_file, 
            depth_file, 
            ply_file,
            color_intrinsics,
            depth_intrinsics,
            extrinsics
        )
        
        try:
            result = detector.process()
            
            if result:
                vis_path = f"result_{rgb_file.name}"
                cv2.imwrite(vis_path, result['visualization'])
                print(f"\n Saved visualization to {vis_path}")
                
                results.append({
                    'image_filename': rgb_file.name,
                    'x': result['center'][0],
                    'y': result['center'][1],
                    'z': result['center'][2],
                    'Rx': result['rotation'][0],
                    'Ry': result['rotation'][1],
                    'Rz': result['rotation'][2]
                })
        except Exception as e:
            print(f"\n✗ Error processing {rgb_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"SAVING RESULTS TO {output_csv}")    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_filename', 'x', 'y', 'z', 'Rx', 'Ry', 'Rz'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n Successfully processed {len(results)}/{len(rgb_files)} images!")
    print(f" Results saved to {output_csv}")
    
    return results

if __name__ == "__main__":
    color_intrinsics, depth_intrinsics, extrinsics = load_camera_params()
    
    detector = ParcelDetector(
        "test/rgb/0266.png",
        "test/depth/0266.png", 
        "test/ply/0266.ply",
        color_intrinsics,
        depth_intrinsics,
        extrinsics
    )
    
    result = detector.process()
    
    if result:
        cv2.imwrite("test_result.png", result['visualization'])
        print("Saved test result!")