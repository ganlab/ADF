import numpy as np
import os
from scipy.spatial import cKDTree
from tqdm import tqdm
import open3d as o3d
import cupy as cp
from PIL import Image
import re
import argparse

def autofill(point_cloud_label, points):
    # Create a mask for black points
    black_mask = np.linalg.norm(point_cloud_label - [0], axis=1) == 0
    black_points = points[black_mask]
    black_label = point_cloud_label[black_mask]

    # Create a mask for color points
    color_mask = ~black_mask
    color_label = point_cloud_label[color_mask]

    # Build a KD-tree
    kdtree = cKDTree(points[color_mask, :3])

    b_len = 0
    eps = 0.05
    while len(black_label) > 0:
        c_len = len(black_label)
        if b_len == c_len:
            eps = eps * 2
            print(f"new eps:{eps}")

        print(f"black_points len:{len(black_label)}")

        # For each black point, search for the nearest non-black neighbor using KD-tree
        for i, point in tqdm(enumerate(black_points)):
            dist, idx = kdtree.query(point[:3], k=1)
            nearest_pt = color_label[idx]
            point_label = nearest_pt

            if dist > eps:
                black_label[i] = [0]
            else:
                black_label[i] = point_label

        # Update the color information
        point_cloud_label[black_mask] = black_label
        black_mask = np.linalg.norm(point_cloud_label - [0], axis=1) == 0
        black_points = points[black_mask]
        black_label = point_cloud_label[black_mask]
        color_mask = ~black_mask
        color_label = point_cloud_label[color_mask]
        kdtree = cKDTree(points[color_mask, :3])
        b_len = c_len

    return point_cloud_label

def extract_camera_parameters(camera_param):
    rotation_matrix = cp.asarray(camera_param.extrinsic[:3, :3])
    translation_vector = cp.asarray(camera_param.extrinsic[:3, 3:4].T)
    intrinsic_matrix = cp.asarray(camera_param.intrinsic.intrinsic_matrix)

    return rotation_matrix, translation_vector, intrinsic_matrix

def extract_intrinsics(matrix):
    fx = matrix[0, 0]
    fy = matrix[1, 1]
    cx = matrix[0, 2]
    cy = matrix[1, 2]
    return fx, fy, cx, cy

def rek_camera_params(camera_params, target_width, target_height):
    # Extract original parameters
    orig_width = camera_params.intrinsic.width
    orig_height = camera_params.intrinsic.height

    rotation_matrix, translation_vector, intrinsic_matrix = extract_camera_parameters(camera_params)
    fx, fy, cx, cy = extract_intrinsics(intrinsic_matrix)

    # Calculate the scale factor for size change
    scale_width = target_width / orig_width
    scale_height = target_height / orig_height

    camera_params.intrinsic.set_intrinsics(
        target_width, target_height,
        fx * scale_width,  # Scaled fx
        fy * scale_height,  # Scaled fy
        cx * scale_width,  # Scaled cx
        cy * scale_height  # Scaled cy
    )

    return camera_params

def check_z_value(points):
    z_values = points[:, 2]  # Get the z coordinates of all points
    result = z_values >= 0  # Check if the z values are greater than or equal to 0
    return result

def assign_labels_to_points(pixel_index_all_points, mask, point_cloud_label):
    height, width= pixel_index_all_points.shape

    for y in range(height):
        for x in range(width):
            point_index = pixel_index_all_points[y, x]
            label = mask[y, x]

            # Update the label value if the label of the corresponding 3D point has not been assigned or the previous label was 0
            if point_cloud_label[point_index] == 0:
                point_cloud_label[point_index] = label
    return point_cloud_label

def parse_resolution(resolution):
    try:
        width, height = map(int, resolution.strip('[]').split(','))
        return [width, height]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid resolution format: {resolution}. Expected format: [width,height]")

def parse_number_list(number_list):
    try:
        return list(map(float, number_list.strip('[]').split(',')))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid number list format: {number_list}. Expected format: [num1,num2,...]")

def main(args):
    # base_folder_path
    origin_path = args.path
    # value_β
    k = [3.2]

    # base_folder_path
    data_names = os.listdir(origin_path)
    for data_name in data_names:
        folder_path = os.path.join(origin_path, data_name)

        # view_parameter_path
        params_path = os.path.join(folder_path, 'params')
        data_name = re.sub(r"_yes$", "", data_name)

        # Load the point cloud data,The point cloud name and the data name are the same
        pcd_name = data_name + '.npy'
        pcd_path = os.path.join(folder_path, pcd_name)
        pcd = cp.load(pcd_path)
        pcd_np = cp.asnumpy(pcd)

        k_names = [f for f in os.listdir(os.path.join(folder_path, 'mask')) if os.path.isdir(os.path.join(os.path.join(folder_path, 'mask'), f))]
        for k_name in k_names:
            point_cloud_label = np.zeros((len(pcd), 1))
            save_path = os.path.join(folder_path, 'mask', k_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            for filename in os.listdir(params_path):
                name, extension = os.path.splitext(filename)
                print(data_name + "_" + name)

                # Load the pixel label value
                mask_path = os.path.join(folder_path, 'mask', name + '.npy')
                mask = np.load(mask_path)

                # Load the correspondence between pixel and 3D point index within the view Angle
                pixel_to_points_index_name = name + '_pixel_to_points_index.npy'
                pixel_index_all_points = np.load(
                    os.path.join(folder_path, 'pixel_to_points_index', k_name, pixel_to_points_index_name)).T

                point_cloud_label = assign_labels_to_points(pixel_index_all_points, mask, point_cloud_label)
                np.savetxt(save_path + '/' + name + '.txt', point_cloud_label, fmt='%d')

            # Fill label values
            point_cloud_label = autofill(point_cloud_label, pcd_np)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            np.savetxt(save_path + '/label.txt', point_cloud_label, fmt='%d')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image generation")
    parser.add_argument('--path', type=str, help='Base folder path', default='./examples')
    args = parser.parse_args()
    main(args)
