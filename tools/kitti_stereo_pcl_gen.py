import os
import cv2
import numpy as np
from tqdm import tqdm


def read_calib_file(calib_path):
    """
    Reads a KITTI calibration file.
    """
    calib_data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            key, value = line.split(':', 1)
            calib_data[key.strip()] = np.array(list(map(float, value.split())))
    return calib_data


def generate_disparity_map(left_img, right_img):
    """
    Generates the disparity map using OpenCV's StereoSGBM.
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Must be divisible by 16
        blockSize=9,
        P1=8 * 3 * 3**2,
        P2=32 * 3 * 3**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    return disparity


def filter_points_by_range(points_3d, point_cloud_range):
    """
    Filters points based on the specified range.
    
    Args:
        points_3d (np.ndarray): Array of 3D points (N x 4 including intensity).
        point_cloud_range (list): [xmin, ymin, zmin, xmax, ymax, zmax].
        
    Returns:
        np.ndarray: Filtered 3D points within the range.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range
    mask = (
        (points_3d[:, 0] >= xmin) & (points_3d[:, 0] <= xmax) &
        (points_3d[:, 1] >= ymin) & (points_3d[:, 1] <= ymax) &
        (points_3d[:, 2] >= zmin) & (points_3d[:, 2] <= zmax)
    )
    return points_3d[mask]


def reproject_to_3d(disparity, Q, point_cloud_range, vertical_step=0.04):
    """
    Reprojects disparity to 3D points, simulates LiDAR scanlines, and applies range constraints.

    Args:
        disparity (np.ndarray): Disparity map.
        Q (np.ndarray): Reprojection matrix.
        point_cloud_range (list): [xmin, ymin, zmin, xmax, ymax, zmax].
        vertical_step (float): Vertical step in meters for simulating LiDAR scanlines.

    Returns:
        np.ndarray: Filtered and transformed point cloud (N x 4, including intensity).
    """
    # Step 1: Reproject disparity to 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0  # Valid disparity mask
    points_3d = points_3d[mask]
    
    # Step 2: Simulate LiDAR scanlines (prune points)

    # Step 2.a: slightly rotate the points downward
    T = np.array([
        [1, 0, 0],
        [0, np.cos(0.1), -np.sin(0.1)],
        [0, np.sin(0.1), np.cos(0.1)]
    ])
    points_3d = points_3d @ T.T

    # Step 2.b: Keep only points that are very close to a plane with Z = k * vertical_step
    points_3d = points_3d[np.abs(points_3d[:, 1] % vertical_step) < 0.01]

    # Step 2.c: rotate the points back
    T = np.array([
        [1, 0, 0],
        [0, np.cos(-0.1), -np.sin(-0.1)],
        [0, np.sin(-0.1), np.cos(-0.1)]
    ])
    points_3d = points_3d @ T.T

    # Step 3: Transform to KITTI's coordinate system
    transformation_matrix = np.array([
        [0, 0, -1],  # Camera X -> KITTI -Z
        [1, 0,  0],  # Camera Y -> KITTI X
        [0, 1,  0],  # Camera Z -> KITTI Y
    ], dtype=np.float32)
    points_3d = points_3d @ transformation_matrix.T

    # Step 4: Add dummy intensity
    intensities = np.ones((points_3d.shape[0], 1), dtype=np.float32)
    points_3d_with_intensity = np.hstack((points_3d, intensities))

    # Step 5: Constrain point cloud range
    points_3d_with_intensity = filter_points_by_range(points_3d_with_intensity, point_cloud_range)

    return points_3d_with_intensity


def process_kitti_dataset(args):
    """
    Processes a KITTI dataset directory and generates .bin point cloud files.
    """

    kitti_path = args.kitti_path
    max_samples = args.max_samples
    image_dir = os.path.join(kitti_path, 'image_2')
    calib_dir = os.path.join(kitti_path, 'calib')
    stereo_pcl_dir = os.path.join(kitti_path, 'stereo_pcl')

    os.makedirs(stereo_pcl_dir, exist_ok=True)

    image_files = sorted(os.listdir(image_dir))
    calib_files = sorted(os.listdir(calib_dir))

    for idx, (img_file, calib_file) in enumerate(tqdm(zip(image_files, calib_files), total=len(image_files))):
        if max_samples is not None and idx >= max_samples:
            break
        if not img_file.endswith('.png') or not calib_file.endswith('.txt'):
            continue

        # Load left and right images
        img_idx = img_file.split('.')[0]
        left_img_path = os.path.join(kitti_path, 'image_2', img_file)
        right_img_path = os.path.join(kitti_path, 'image_3', img_file)

        left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

        # Load calibration
        calib = read_calib_file(os.path.join(calib_dir, calib_file))
        P2 = calib['P2'].reshape(3, 4)  # Projection matrix for left camera
        P3 = calib['P3'].reshape(3, 4)  # Projection matrix for right camera

        # Compute baseline and Q matrix
        f = P2[0, 0]  # Focal length
        cx, cy = P2[0, 2], P2[1, 2]  # Principal points
        baseline = abs((P2[0, 3] - P3[0, 3]) / f)  # Baseline

        Q = np.array([[1, 0, 0, -cx],
                      [0, 1, 0, -cy],
                      [0, 0, 0, f],
                      [0, 0, -1 / baseline, 0]])

        # Compute disparity and reproject to 3D
        # print(f"Processing image {img_idx}...")
        disparity = generate_disparity_map(left_img, right_img)
        points_3d = reproject_to_3d(disparity, Q, point_cloud_range=[0, -40, -3, 70, 40, 1])

        # Save to .bin
        output_path = os.path.join(stereo_pcl_dir, f"{img_idx}.bin")
        points_3d.astype(np.float32).tofile(output_path)

    print(f"Point clouds saved to {stereo_pcl_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate point clouds from KITTI stereo dataset.")
    parser.add_argument(
        "kitti_path",
        type=str,
        help="Path to KITTI dataset subdirectory (e.g., data/KITTI/training)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    args = parser.parse_args()

    process_kitti_dataset(args)
