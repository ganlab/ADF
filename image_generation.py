import numpy as np
import cupy as cp
import os
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import time
from scipy.spatial import KDTree
import argparse
def caculate_z_boundary_value(r, intrinsic_matrix):
    fx, fy, cx, cy = extract_intrinsics(intrinsic_matrix)
    z = r * fx
    return z

def calculate_d(Z_delta, z_boundary, s = 1):
    numerator = Z_delta * s**2
    denominator = 2 * z_boundary - 2 * s * Z_delta
    with np.errstate(divide='ignore', invalid='ignore'):
        d_array = np.where(denominator != 0, numerator / denominator, np.nan)
    return d_array

def calculate_square_areas_vectorized_diff_sizes(centers, sides, points_Z, width, length, batch_size):
    # Ensure the centers coordinate array is two-dimensional
    centers = cp.atleast_2d(centers)
    half_sides = cp.atleast_1d(sides / 2)

    n = centers.shape[0]

    min_values = cp.full((width, length), cp.inf)
    min_indices = cp.full((width, length), -1)  # Initialize to -1 or any invalid index

    for i in tqdm(range(0, n, batch_size), desc="Processing batches"):
        batch_centers = centers[i:i + batch_size]
        batch_half_sides = half_sides[i:i + batch_size]

        x_grid, y_grid = cp.meshgrid(cp.arange(width), cp.arange(length), indexing='ij')
        centers_exp = batch_centers[:, np.newaxis, np.newaxis, :]
        half_sides_exp = batch_half_sides[:, np.newaxis, np.newaxis]

        # Initialize in_squares, set the coordinates of centers inside the regions to True
        current_batch_size = batch_centers.shape[0]  # Actual size of the current batch
        in_squares = cp.zeros((current_batch_size, width, length), dtype=cp.bool_)
        # Get x and y coordinates of all center points
        x_coords = cp.round(batch_centers[:, 0]).astype(cp.int32)
        y_coords = cp.round(batch_centers[:, 1]).astype(cp.int32)
        # Confirm coordinates are within the valid range (within (width, length))
        valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < length)
        # Obtain valid indices
        valid_x = x_coords[valid_indices]
        valid_y = y_coords[valid_indices]
        valid_batch_indices = cp.nonzero(valid_indices)[0]  # Keep the dimension for indexing
        # Update the values of in_squares corresponding to coordinates to True
        in_squares[valid_batch_indices, valid_x, valid_y] = True

        # Detect within square areas
        distance_x = cp.abs(x_grid - centers_exp[..., 0])
        distance_y = cp.abs(y_grid - centers_exp[..., 1])
        in_squares |= (distance_x <= half_sides_exp) & (distance_y <= half_sides_exp)

        batch_points_Z_matrix = cp.full((batch_centers.shape[0], width, length), cp.inf)

        # Generate a points_Z_expanded array conforming to the shape requirement
        points_Z_expanded = points_Z[i:i + batch_size][:, np.newaxis, np.newaxis]
        points_Z_expanded = cp.broadcast_to(points_Z_expanded, in_squares.shape)

        # Use in_squares boolean array as a mask to update the values of batch_points_Z_matrix
        batch_points_Z_matrix[in_squares] = points_Z_expanded[in_squares]

        batch_min_values = cp.min(batch_points_Z_matrix, axis=0)

        # Compare the current batch minimum values with previous minimum values
        update_mask = batch_min_values < min_values
        min_values[update_mask] = batch_min_values[update_mask]

        batch_min_indices = cp.argmin(batch_points_Z_matrix, axis=0) + i  # Add i offset to reflect global indices
        min_indices[update_mask] = batch_min_indices[update_mask]

    return min_indices


def update_negative_indices_with_minZ(pxiel_Z, width, length, side,
                                      min_indices, batch_size):
    # Find the pixel positions with a value of -1 (flattened row-wise)
    negative_indices = cp.where(min_indices.flatten() == -1)[0]

    if negative_indices.size > 0:
        # Convert to two-dimensional coordinates (neg_y---row index (1024), neg_x---column index (576))
        neg_y, neg_x = cp.divmod(negative_indices, length)

        # Create new centers and corresponding sides arrays for pixels with a value of -1
        new_centers = cp.stack((neg_y, neg_x), axis=-1)
        new_sides = cp.full(new_centers.shape[0], side, dtype=new_centers.dtype)  # Assume all centers with -1 have the same side length

        # Calculate the index with the minimum Z value for the new centers
        n = new_centers.shape[0]
        for i in tqdm(range(0, n, batch_size), desc=""):
            batch_centers = new_centers[i:i + batch_size]
            batch_half_sides = new_sides[i:i + batch_size] / 2

            # Generate x_grid and y_grid
            y_grid, x_grid = cp.meshgrid(cp.arange(width), cp.arange(length), indexing='ij')
            centers_exp = batch_centers[:, cp.newaxis, cp.newaxis, :]
            half_sides_exp = batch_half_sides[:, cp.newaxis, cp.newaxis]

            # Create a boolean array to check if a point is within the corresponding square
            distance_x = cp.abs(x_grid - centers_exp[..., 1])
            distance_y = cp.abs(y_grid - centers_exp[..., 0])
            in_squares = (distance_x <= half_sides_exp) & (distance_y <= half_sides_exp)

            # First, expand pixel_Z by a new axis to match the batch size of in_squares, then use broadcasting to compare
            expanded_pixel_Z = pxiel_Z[cp.newaxis, :, :]
            # Use in_squares as a mask to select corresponding values from expanded_pixel_Z, setting non-mask areas to NaN
            batch_Z_values = cp.where(in_squares, expanded_pixel_Z, cp.inf)
            # Find the index of the minimum value (after handling NaN, NaN is replaced with cp.inf)
            flat_indices = cp.argmin(batch_Z_values.reshape(batch_Z_values.shape[0], -1), axis=1)
            # rows---rows  cols---columns
            rows = flat_indices // batch_Z_values.shape[2]  # Convert to two-dimensional row index
            cols = flat_indices % batch_Z_values.shape[2]  # Convert to two-dimensional column index

            # Check if each (i, width, length) is all cp.inf, meaning the area's values are invalid or no minimum exists
            is_inf = cp.all(cp.isinf(batch_Z_values), axis=(1, 2))

            # Replace coordinates of all cp.inf cases with coordinates from batch_centers
            rows[is_inf] = batch_centers[is_inf, 0]
            cols[is_inf] = batch_centers[is_inf, 1]

            # First, split new_centers' coordinates into x and y coordinate arrays
            neg_y, neg_x = batch_centers[:, 0], batch_centers[:, 1]

            # Use rows and cols to index min_indices, getting the corresponding 3D point indices
            corresponding_3d_point_indices = min_indices[rows, cols]

            # Directly use neg_x and neg_y as indices to batch update the new_min_indices array
            # Note we use neg_y and neg_x as indices to conform to row and column arrangement
            min_indices[neg_y, neg_x] = corresponding_3d_point_indices

    return min_indices

def filter_points_and_indices(projected_points, width, length):
    indices = cp.where((projected_points[:, 0] >= 0) & (projected_points[:, 0] <= width) &
                       (projected_points[:, 1] >= 0) & (projected_points[:, 1] <= length))[0]
    filtered_points = projected_points[indices]
    return indices, filtered_points

def extract_camera_parameters(camera_param):
    # Extract rotation matrix
    rotation_matrix = cp.asarray(camera_param.extrinsic[:3, :3])
    # Extract translation vector and transpose
    translation_vector = cp.asarray(camera_param.extrinsic[:3, 3:4].T)
    # Extract intrinsic matrix
    intrinsic_matrix = cp.asarray(camera_param.intrinsic.intrinsic_matrix)

    return rotation_matrix, translation_vector, intrinsic_matrix

def check_z_value(points):
    z_values = points[:, 2]  # Get z coordinates of all points
    result = z_values >= 0  # Check if z values are greater than or equal to 0
    return result

def Compute_image_color(point_cloud, camera_param, width, length, radius, batch_size):
    # Extracting camera parameters
    rotation_matrix, translation_vector, intrinsic_matrix = extract_camera_parameters(camera_param)

    # Change rgb range to [0-255]
    mask = point_cloud[:, 3:6][0] < 1
    if mask[0]:
        point_cloud[:, 3:6] = point_cloud[:, 3:6] * 255

    # pixel_rgb --- rgb values corresponding to each pixel
    # pixel_rgb_expan --- pixel_rgb expanded
    pixel_rgb = cp.ones((length, width, 3), dtype=cp.float64) * 255
    pixel_rgb_expan = cp.ones((length, width, 3), dtype=cp.float64) * 255

    # The point cloud coordinates are transformed from the world coordinates to the camera coordinate system
    # point_cloud_in_view --- Points in the field of view(In the world coordinate system)
    # transformed_point_in_view_cloud --- Points in the field of view(In camera coordinates)
    transformed_point_cloud = cp.dot(point_cloud[:, :3], rotation_matrix.T) + translation_vector
    points_in_view_index = check_z_value(transformed_point_cloud)
    in_view_index_to_all = np.where(points_in_view_index)[0]
    point_cloud_in_view = point_cloud[points_in_view_index.flatten()]
    transformed_point_in_view_cloud = transformed_point_cloud[points_in_view_index.flatten()]
    # Projection to the image plane
    projected_points = cp.dot(intrinsic_matrix, transformed_point_in_view_cloud.T).T
    projected_points /= projected_points[:, 2].reshape(-1, 1)

    # Gets the index of the coordinates in the image range
    in_image_pixel_indices, in_image_pixel = filter_points_and_indices(projected_points[:, :2], width, length)
    in_image_points = transformed_point_in_view_cloud[in_image_pixel_indices]

    # calculate z_boundary
    z_boundary = caculate_z_boundary_value(radius, intrinsic_matrix)
    
    # Computing depth values
    z_delta_array = cp.asarray([z_boundary]) - in_image_points[:, 2]
    d_array = calculate_d(z_delta_array, z_boundary, 1)

    # pixel_to_point --- The index of the corresponding point of the pixel
    pixel_to_point = calculate_square_areas_vectorized_diff_sizes(in_image_pixel, d_array,
                                                          in_image_points[:, 2], width, length, batch_size)

    # calculate pixel_rgb
    valid_mask = pixel_to_point != -1
    valid_indices = np.where(valid_mask.flatten())[0]
    point_indices = pixel_to_point.flatten()[valid_indices]
    rgb_values = point_cloud_in_view[in_image_pixel_indices[point_indices], 3:6]
    rows, cols = np.divmod(valid_indices, pixel_to_point.shape[1])
    pixel_rgb[cols, rows] = rgb_values

    # pixel_Z --- An array of z-values corresponding to each pixel is recorded
    pixel_Z = cp.full_like(pixel_to_point, cp.inf, dtype=cp.float64)
    pixel_Z[valid_mask] = in_image_points[pixel_to_point[valid_mask], 2]

    side = 4
    pixel_to_point_expan = pixel_to_point.copy()
    pixel_to_point_expan = update_negative_indices_with_minZ(pixel_Z, width, length, side, pixel_to_point_expan, batch_size)

    valid_mask = pixel_to_point_expan != -1
    valid_indices = np.where(valid_mask.flatten())[0]
    point_indices = pixel_to_point_expan.flatten()[valid_indices]
    rgb_values = point_cloud_in_view[in_image_pixel_indices[point_indices], 3:6]
    rows, cols = np.divmod(valid_indices, pixel_to_point_expan.shape[1])
    pixel_rgb_expan[cols, rows] = rgb_values

    # Update 3D point index in pixel_to_point (from in_images --> in_view)
    mask = pixel_to_point != -1
    valid_indices = pixel_to_point[mask]
    pixel_to_point[mask] = in_image_pixel_indices[valid_indices]
    # Update index of 3D points in pixel_to_point (from in_view --> all_points)
    valid_indices = pixel_to_point[mask]
    pixel_to_point[mask] = in_view_index_to_all[valid_indices]

    return pixel_rgb, pixel_rgb_expan, pixel_to_point


def compute_average_nearest_neighbor_distance(points, x, k):
    """
    Compute the average distance of each point to its nearest x neighbor points.

    Parameters:
    points : cp.array
        NumPy array of shape nx3 representing point cloud data.
    x : int
        Number of neighbors.

    Returns:
    float
        Average distance of all points to their nearest x neighbors.
    """
    # Create KDTree for input data, ensuring conversion of CuPy arrays to NumPy arrays
    tree = KDTree(points.get())

    # Initialize a list to store average distances for all points
    average_distances = []

    num = len(points) * k
    count = 0
    # Iterate over each point in the point cloud. Here, ensure iteration over NumPy array
    for point_np in points.get():  # Conversion to NumPy array using .get() method here
        # Query the nearest x+1 points (including the point itself)
        distances, _ = tree.query(point_np, k=x + 1)  # Note the usage of the point's NumPy version

        # Remove the distance to the point itself (distance = 0)
        distances = distances[1:]

        # Calculate the average distance and append to the list
        average_distance = cp.mean(cp.asarray(distances))  # Ensure conversion to CuPy array if you want to do computations on GPU
        average_distances.append(average_distance)

        if count > num:
            break
        count += 1
    # Then calculate the average of the average distances of all points to describe the distance between points in the overall point cloud
    overall_average = cp.mean(cp.asarray(average_distances))  # Ensure conversion to CuPy array if you want to do computations on GPU

    return overall_average.get()  # Finally, return the value converted to a NumPy array

def extract_intrinsics(matrix):
    fx = matrix[0, 0]
    fy = matrix[1, 1]
    cx = matrix[0, 2]
    cy = matrix[1, 2]
    return fx, fy, cx, cy

def resize_camera_params(camera_params, target_width, target_height):
    # Extract original parameters
    orig_width = camera_params.intrinsic.width
    orig_height = camera_params.intrinsic.height

    rotation_matrix, translation_vector, intrinsic_matrix = extract_camera_parameters(camera_params)
    fx, fy, cx, cy = extract_intrinsics(intrinsic_matrix)

    # Calculate the scale of size change
    scale_width = target_width / orig_width
    scale_height = target_height / orig_height

    # Update the intrinsic matrix
    # Use the new resolution and scale the focal lengths and principal point coordinates accordingly
    camera_params.intrinsic.set_intrinsics(
        target_width, target_height,
        fx * scale_width,  # Scaled fx
        fy * scale_height,  # Scaled fy
        cx * scale_width,  # Scaled cx
        cy * scale_height  # Scaled cy
    )

    return camera_params

def convert_seconds(duration):
    hours = duration // 3600  # Get the number of hours by integer division
    minutes = (duration % 3600) // 60  # Convert the remaining seconds to minutes
    seconds = duration % 60  # Remaining seconds

    return f"{hours}hour{minutes}minute{seconds}second"

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
    folder_path = args.path
    target_width = args.resolution[0]
    target_height = args.resolution[1]
    # value_β
    k_array = args.k
    batch_size = args.batch
    # get_all_data_name
    data_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    times = []
    for data_name in data_names:
        # data_folder_path
        data_path = os.path.join(folder_path, data_name)
        # view_parameter_path
        view_params_path = os.path.join(data_path, 'params')

        # Load the point cloud data,The point cloud name and the data name are the same
        pcd_name = data_name + '.npy'
        pcd_path = os.path.join(data_path, pcd_name)
        pcd = cp.load(pcd_path)

        # pcd_name = data_names[i] + '.txt'
        # pcd_path = os.path.join(data_path, pcd_name)
        # pcd = cp.loadtxt(pcd_path)

        # The indices of pixels and point clouds at different viewpoints are computed
        for view_param_name in os.listdir(view_params_path):
            # start counting
            start_time = time.time()

            # Read the camera viewpoint parameters
            name, extension = os.path.splitext(view_param_name)
            camera_param = o3d.io.read_pinhole_camera_parameters(os.path.join(view_params_path, name + '.json'))
            # Modify the camera parameters with the goal of changing the generated image resolution
            if [target_width, target_height] != [-1, -1]:
                camera_param = resize_camera_params(camera_param, target_width, target_height)

            # Calculate the point cloud spacing
            r = compute_average_nearest_neighbor_distance(pcd[:, :3], 4, 0.01)

            for x in range(len(k_array)):
                k = k_array[x]
                radius = r * k
                print(data_name + '_param=' + name + '_r=' + str(k))
                width = camera_param.intrinsic.width
                length = camera_param.intrinsic.height

                # Calculate indices of pixels and point clouds
                pixel_rgb, pixel_rgb_expan, pixel_all_point_index = Compute_image_color(pcd, camera_param, width,
                                                                                        length, radius, batch_size)

                pixel_rgb = cp.asnumpy(pixel_rgb)
                pixel_rgb_expan = cp.asnumpy(pixel_rgb_expan)

                pixel_rgb = [pixel_rgb, pixel_rgb_expan]
                name_arrays = ['not_expan', 'generation']

                for idx, image_arr in enumerate(pixel_rgb):

                    image_arr = (image_arr).round().astype(np.uint8)
                    image = Image.fromarray(image_arr)

                    # output information
                    image_generation_out_path = os.path.join(folder_path, 'generation_image')
                    if not os.path.exists(image_generation_out_path):
                        os.mkdir(image_generation_out_path)
                    k_image_generation_out_path = os.path.join(image_generation_out_path, 'β=' + str(k))
                    if not os.path.exists(k_image_generation_out_path):
                        os.mkdir(k_image_generation_out_path)
                    image.save(os.path.join(k_image_generation_out_path, name + f'_{name_arrays[idx]}.png'))

                pixel_to_points_index_path = os.path.join(folder_path, 'pixel_to_points_index')
                if not os.path.exists(pixel_to_points_index_path):
                    os.mkdir(pixel_to_points_index_path)
                k_pixel_to_points_index_path = os.path.join(pixel_to_points_index_path, 'β=' + str(k))
                if not os.path.exists(k_pixel_to_points_index_path):
                    os.mkdir(k_pixel_to_points_index_path)
                np.save(os.path.join(k_pixel_to_points_index_path, name + '_pixel_to_points_index.npy'),
                        pixel_all_point_index)

                end_time = time.time()
                duration = end_time - start_time
                formatted_duration = convert_seconds(duration)
                time_path = os.path.join(folder_path, 'time_count')
                if not os.path.exists(time_path):
                    os.mkdir(time_path)
                k_time_path = os.path.join(time_path, 'β=' + str(k))
                if not os.path.exists(k_time_path):
                    os.mkdir(k_time_path)
                with open(os.path.join(k_time_path, 'times_output.txt'), 'a') as f:
                    f.write(f"{name}: {formatted_duration}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image generation")
    parser.add_argument('--path', type=str, help='Base folder path', default='./examples')
    parser.add_argument('--k', type=parse_number_list, help='List of β [n1,n2,...]', default=[3.2])
    parser.add_argument('--resolution', type=parse_resolution, help='Target resolution as [width,height]', default=[-1, -1])
    parser.add_argument('--batch', type=int, help='gpu batch size', default=200)
    args = parser.parse_args()
    main(args)
