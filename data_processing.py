"""
data_processing.py

Author: James Daniels

"""
import open3d as o3d
import os
import numpy as np
from pathlib import Path
from utils import draw_registration_result, overlap_percentage
import re
import pandas as pd
from utils import calculate_total_points
import copy


def preprocess_point_clouds(dataset_name: str, 
                            overlap_threshold: float = 0.4) -> None:
    """
    Preprocesses the point cloud data in a directory and calculates overlap percentages.

    The function first checks the dataset name to determine the type of dataset. If the 
    dataset name starts with 'sun3d', it calls the 'preprocess_sun3d_point_clouds' function;
    otherwise, it checks for 'eth' or 'cross-source' and calls the appropriate function.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset, used to construct the path to the directory containing 
        the point cloud files.
    overlap_threshold : float, optional
        The overlap threshold for saving point cloud pairs to a text file. 
        The default is 0.4.

    Returns
    -------
    None.
    """
    if dataset_name.lower().startswith('sun3d'):
        preprocess_sun3d_point_clouds(dataset_name, overlap_threshold)

    elif dataset_name.lower().startswith('cross-source'):
        preprocess_cross_source_point_clouds(dataset_name, overlap_threshold)

    else:
        preprocess_eth_point_clouds(dataset_name, overlap_threshold)


def preprocess_cross_source_point_clouds(dataset_name: str, 
                                         overlap_threshold: float = 0.4) -> None:
    """
    Preprocesses the point cloud data in a directory and calculates overlap percentages.
    
    Parameters
    ----------
    dataset_name : str
        The name of the dataset, used to construct the path to the directory containing 
        the point cloud files.
    overlap_threshold : float, optional
        The overlap threshold for saving point cloud pairs to a text file. 
        The default is 0.4.

    Returns
    -------
    None.
    """
    # Construct the path to the directory containing the point cloud files
    directory = Path(".") / "data/cross-source-dataset"

    # Define the file path for the overlapping point clouds file
    overlap_file_path = directory / f"overlapping_point_clouds_overlap_{overlap_threshold}.txt"

    directory.mkdir(parents=True, exist_ok=True)

    # Check if the overlap file already exists
    if overlap_file_path.is_file():
        print(f"Overlap file {overlap_file_path} already exists. Skipping preprocessing.")
        return

    # Define the point cloud pairs
    point_cloud_pairs = []

    # Walk through the kinect_lidar directory
    # for root, dirs, files in os.walk(directory / 'kinect_lidar'):
    #     if 'kinect.ply' in files and 'lidar.ply' in files:
    #         point_cloud_pairs.append((os.path.join(root, 'kinect.ply'), os.path.join(root, 'lidar.ply')))

    for root, dirs, files in os.walk(directory / 'kinect_lidar'):
        if 'kinect.ply' in files and 'lidar.ply' in files:
            kinect_path = Path(root) / 'kinect.ply'
            lidar_path = Path(root) / 'lidar.ply'
            point_cloud_pairs.append((str(kinect_path.relative_to(directory)), str(lidar_path.relative_to(directory))))

    # Walk through the kinect_sfm directory
    for difficulty in ['easy', 'hard']:
        for root, dirs, files in os.walk(directory / 'kinect_sfm' / difficulty):
            if 'kinect.ply' in files and 'sfm.ply' in files:
                kinect_path = Path(root) / 'kinect.ply'
                sfm_path = Path(root) / 'sfm.ply'
                point_cloud_pairs.append((str(kinect_path.relative_to(directory)), str(sfm_path.relative_to(directory))))

    # for difficulty in ['easy', 'hard']:
    #     for root, dirs, files in os.walk(directory / 'kinect_sfm' / difficulty):
    #         if 'kinect.ply' in files and 'sfm.ply' in files:
    #             point_cloud_pairs.append((os.path.join(root, 'kinect.ply'), os.path.join(root, 'sfm.ply')))

    # Process the point cloud pairs
    data_to_write = []  # This will hold the data to write to the file
    for filename1, filename2 in point_cloud_pairs:
        # # Read the point cloud files
        # pcd1 = o3d.io.read_point_cloud(filename1)
        # pcd2 = o3d.io.read_point_cloud(filename2)
        # Read the point cloud files
        pcd1 = o3d.io.read_point_cloud(str(directory / filename1))
        pcd2 = o3d.io.read_point_cloud(str(directory / filename2))


        # print(filename1)
        # o3d.visualization.draw_geometries([pcd1],width=1200,height=800,left=400,top=150,window_name="before color")
        # # Assigning color to the point clouds

        # colors = np.array([[1, 0, 0]] * len(pcd1.points))  # Create color array
        # pcd1.colors = o3d.utility.Vector3dVector(colors)  # Assign colors to the point cloud

        # o3d.visualization.draw_geometries([pcd1],width=1200,height=800,left=400,top=150,window_name="after color")
        # pcd2.paint_uniform_color([0, 1, 0])  # Green for pcd2

        # Read the transformation matrix from the T_gt.txt file
        # transformation_matrix = np.loadtxt(os.path.join(os.path.dirname(filename1), 'T_gt.txt'))
        # Read the transformation matrix from the T_gt.txt file
        transformation_matrix = np.loadtxt(directory / filename1.rsplit('/', 1)[0] / 'T_gt.txt')


        # Show point clouds before applying ground truth transformation
        # draw_registration_result(pcd1, pcd2, np.identity(4), name=f"Before transformation")

        # Show point clouds after applying ground truth transformation to second point cloud
        # draw_registration_result(pcd2, pcd1, transformation_matrix, name=f"After transformation")

        # need to apply the ground truth here to calculate the overlap...

        pcd2 = pcd2.transform(transformation_matrix)

        # Show point clouds before applying ground truth transformation
        # draw_registration_result(pcd1, pcd2, np.identity(4), name=f"same system check")

        # Calculate the overlap percentage
        overlap = overlap_percentage(pcd1, pcd2)

        print(overlap)

        if overlap == 0:
            print(filename1)
            print(filename2)
            # draw_registration_result(pcd1, pcd2, np.identity(4), name=f"same system check")

        # If the overlap percentage is above the threshold, save the pair of point clouds and their corresponding ground truth transformation matrix
        if overlap > overlap_threshold:
            total_points = calculate_total_points(pcd1, pcd2)
            # Add the data to write to the list
            # Instead of using id, use len(data_to_write) to set the id
            data_to_write.append(f"{len(data_to_write)},{filename1},{filename2},{overlap},{total_points}," + ",".join(map(str, transformation_matrix.flatten())))
            # data_to_write.append(f"{id},{filename1},{filename2},{overlap},{total_points}," + ",".join(map(str, transformation_matrix.flatten())))

    # Now, write all the data to the file
    with open(overlap_file_path, "w") as file:
        # Write the header
        file.write("id,filename1,filename2,overlap,total_points," + ",".join([f"transform_{i}" for i in range(16)]) + "\n")
        # Write the data
        for line in data_to_write:
            file.write(line + "\n")

    print("Preprocessing complete.")

def preprocess_sun3d_point_clouds(dataset_name: str, 
                                  overlap_threshold: float = 0.4) -> None:
    """
    Preprocesses the point cloud data in a directory and calculates overlap percentages.

    The function reads the pairs of point clouds from the gt.log file in the given 
    directory, converts them to Open3D point clouds, and calculates the overlap 
    percentage between each pair of point clouds. The overlap percentages are stored 
    in a square matrix, where the entry at row i and column j represents the 
    overlap percentage between the i-th and j-th point clouds.

    If the overlap percentage is above a specified threshold, the pair of point clouds 
    is saved to a text file, alongside their corresponding ground truth transformation matrix.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset, used to construct the path to the directory containing 
        the point cloud files.
    overlap_threshold : float, optional
        The overlap threshold for saving point cloud pairs to a text file. 
        The default is 0.5.

    Returns
    -------
    None.
    """

    # Construct the path to the directory containing the point cloud files
    directory = Path(".") / "data/SUN3D" / dataset_name

    # Define the file path for the overlap matrix file
    overlap_matrix_path = directory / f"overlap_matrix.csv"

    # Define the file path for the overlapping point clouds file
    overlap_file_path = directory / f"overlapping_point_clouds_overlap_{overlap_threshold}.txt"

    # Check if the overlap file already exists
    if overlap_file_path.is_file():
        print(f"Overlap file {overlap_file_path} already exists. Skipping preprocessing.")
        return

    # Read the log file
    with open(directory / "gt.log") as f:
        log_lines = f.readlines()

    # Initialize the overlap matrix
    matrix_size = int(log_lines[0].split()[2])
    overlap_matrix = np.zeros((matrix_size, matrix_size))

    # Initialize list to hold point clouds and total number of points
    point_clouds = []
    total_points_list = []  # list to store total points for each pair of point clouds
    # Initialize the transformation matrices dictionary
    transformation_matrices = {}


    # Iterate over every 5 lines in the log file (since each entry is 5 lines long)
    for i in range(0, len(log_lines), 5):
        # Parse the indices of the point clouds from the first line
        index1, index2 = map(int, log_lines[i].split()[:2])

        # Parse the transformation matrix from the next 4 lines
        transformation_matrix = np.array([list(map(float, line.split())) for line in log_lines[i+1:i+5]])

        # Store the transformation matrix in the dictionary
        transformation_matrices[(index1, index2)] = transformation_matrix

        # Read the point cloud files
        pcd1 = o3d.io.read_point_cloud(str(directory / f"cloud_bin_{index1}.ply"))
        pcd2 = o3d.io.read_point_cloud(str(directory / f"cloud_bin_{index2}.ply"))

        # Show point clouds before applying ground truth transformation
        # draw_registration_result(pcd1, pcd2, np.identity(4), name=f"Before transformation {index1}-{index2}")

        # Show point clouds after applying ground truth transformation
        # draw_registration_result(pcd2, pcd1, transformation_matrix, name=f"After transformation {index1}-{index2}")

        # Append the point clouds to the list
        point_clouds.extend([pcd1, pcd2])

        pcd2 = pcd2.transform(transformation_matrix)

        # Show point clouds before applying ground truth transformation
        # draw_registration_result(pcd1, pcd2, np.identity(4), name=f"same coord system?")

        overlap = overlap_percentage(pcd1, pcd2)
        overlap_matrix[index1, index2] = overlap
        overlap_matrix[index2, index1] = overlap

        # Calculate number of points
        number_of_points = calculate_total_points(pcd1, pcd2)
        total_points_list.append(number_of_points)

    # Create a DataFrame from the overlap matrix
    overlap_df = pd.DataFrame(overlap_matrix)

    # Save the overlap matrix to a CSV file
    overlap_df.to_csv(overlap_matrix_path)

    # Save pairs of point clouds with overlap above the threshold, and their corresponding ground truth transformation matrix
    with open(overlap_file_path, "w") as file:
        # Include a new column for total number of points in the header
        file.write("id,filename1,filename2,overlap,total_points," + ",".join([f"transform_{i}" for i in range(16)]) + "\n")
        id = 0
        for i in range(overlap_matrix.shape[0]):
            for j in range(i+1, overlap_matrix.shape[1]):
                if overlap_matrix[i, j] > overlap_threshold:
                    transformation_matrix_flat = transformation_matrices[(i, j)].flatten()
                    total_points = total_points_list[id]  # Get the total number of points for this pair of point clouds
                    # Include the total number of points in the written line
                    file.write(f"{id},cloud_bin_{i}.ply,cloud_bin_{j}.ply,{overlap_matrix[i, j]},{total_points}," + ",".join(map(str, transformation_matrix_flat)) + "\n")
                    id += 1

    print("Preprocessing complete.")


def preprocess_eth_point_clouds(dataset_name: str, 
                                overlap_threshold: float = 0.4) -> None:
        # Construct the path to the directory containing the point cloud files
    directory = Path(".") / "data/ETH" / dataset_name

    # Define the file path for the overlap matrix file
    overlap_matrix_path = directory / f"overlap_matrix_{overlap_threshold}.csv"

    # Define the file path for the overlapping point clouds file
    overlap_file_path = directory / f"overlapping_point_clouds_overlap_{overlap_threshold}.txt"

    # Check if the overlap file already exists
    if overlap_file_path.is_file():
        print(f"Overlap file {overlap_file_path} already exists. Skipping preprocessing.")
        return
    
    # Get a list of all point cloud files in the directory
    filenames = [f for f in os.listdir(directory) if f.startswith('PointCloud') and f.endswith('.csv')]

    # Initialize list to hold point clouds
    point_clouds = []

    # Read each file and convert to an open3d point cloud
    for i, filename in enumerate(filenames):
        # Read the file into a pandas DataFrame
        df = pd.read_csv(os.path.join(directory, filename))

        # Convert the DataFrame to an open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
        point_clouds.append(pcd)

    print(f"Read {len(point_clouds)} point clouds.")

    # Check if the overlap matrix file already exists
    if overlap_matrix_path.is_file():
        # Load the overlap matrix from the file
        overlap_df = pd.read_csv(overlap_matrix_path, index_col=0)
        overlap_matrix = overlap_df.values
    else:
        # Initialize the overlap matrix
        overlap_matrix = np.zeros((len(filenames), len(filenames)))

        # Calculate overlap percentages and save them in the overlap matrix
        for i in range(len(point_clouds)):
            for j in range(i+1, len(point_clouds)):
                # Inside the overlap computation loop
                # print(f"Calculating overlap between point cloud {i} and {j}...")
                overlap = overlap_percentage(point_clouds[i], point_clouds[j])
                # print(overlap)
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap

        # Create a DataFrame from the overlap matrix and add filenames as row and column labels
        overlap_df = pd.DataFrame(overlap_matrix, columns=filenames, index=filenames)

        # Save the overlap matrix to a CSV file
        overlap_df.to_csv(overlap_matrix_path)

    with open(overlap_file_path, "w") as file:
        file.write("id,filename1,filename2,overlap,total_points\n")  # add total_points to the header
        id = 0
        for i in range(len(filenames)):
            for j in range(i+1, len(filenames)):
                if overlap_matrix[i, j] > overlap_threshold:
                    total_points = calculate_total_points(point_clouds[i], point_clouds[j])
                    file.write(f"{id},{filenames[i]},{filenames[j]},{overlap_matrix[i, j]},{total_points}\n")  # write total_points to each row
                    id += 1
    print("Preprocessing complete.")

