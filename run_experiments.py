"""
run_experiments.py

Author: James Daniels

"""
import torch
import open3d as o3d
import numpy as np
import time
import os
from pathlib import Path
from utils import load_point_cloud, save_transformations, draw_registration_result, calculate_total_points
from transformation import random_transformation_matrix
from evaluation_metrics import get_full_evaluation_metrics, save_full_eval_metrics
from registration_algorithms import downsample_point_cloud
import inspect
from typing import Callable


def run_all_experiments(dataset_names: list, 
                        algorithms: list, 
                        voxelsize: float, 
                        range_t: float, 
                        range_r: float, 
                        overlap: float):
    """
    Run an algorithm on a dataset and save the metrics.

    Parameters:
    dataset_names: List of dataset names to run the experiments on.
    algorithms: List of algorithms to be used for registration.
    voxelsize: Size of voxel to be used in the point cloud.
    range_t: Range for generating transformation matrix.
    range_r: Range for generating rotation matrix.
    overlap: Percentage overlap between point clouds.
    """

    for algorithm in algorithms:
        for dataset_name in dataset_names:
            # Define path to the file where the results will be stored
            metrics_file_path = f"./results/{dataset_name}/{algorithm.__name__}/voxelsize{voxelsize}_rangeT{range_t}_rangeR{range_r}_overlap{overlap}_metrics.txt"

            # Checking file path here
            # If the metrics file already exists, skip the experiment
            if os.path.exists(metrics_file_path):
                print(f"Metrics file {metrics_file_path} already exists. Skipping the experiment.")

            else:
                print(f"Running {algorithm.__name__} algorithm on {dataset_name} dataset...")

                main_run_experiments(algorithm, dataset_name, voxelsize, range_t, range_r, overlap, metrics_file_path)


def main_run_experiments(algorithm, 
                         dataset_name: str, 
                         voxelsize: float, 
                         range_t: float, 
                         range_r: float, 
                         overlap: float,
                         metrics_file_path: str):
    """
    Runs the specified algorithm on the dataset and saves the results.

    Parameters:
    algorithm: The registration algorithm.
    dataset_name: The name of the dataset.
    voxelsize: Size of voxel to be used in the point cloud.
    range_t: Range for generating transformation matrix.
    range_r: Range for generating rotation matrix.
    metrics_file_path: Path where metrics file will be saved.
    """

    # Determine the dataset folder based on the dataset name
    if dataset_name.lower().startswith('sun3d'):
        dataset_folder = Path(".") / "data/SUN3D" / dataset_name
    elif dataset_name.lower().startswith('cross-source'):
        dataset_folder = Path(".") / "data/cross-source-dataset"
    else:
        dataset_folder = Path(".") / "data/ETH" / dataset_name

    # List to store evaluation metrics
    full_metrics_list = []

    # Open and iterate over the overlapping_point_clouds.txt file
    with open(dataset_folder / f"overlapping_point_clouds_overlap_{overlap}.txt", "r") as f:
        next(f)  # Skip the header line

        # Loop over all lines in the file
        for line in f:
            # Split the line into components
            components = line.strip().split(',')
            # id, filename1, filename2, overlap = components[:4]
            id, filename1, filename2, overlap, total_points = components[:5]  # Added total_points here

            # Load the point clouds
            pcd_0 = load_point_cloud(dataset_folder / filename1)
            pcd_1 = load_point_cloud(dataset_folder / filename2)

            # # Apply the ground truth transformation matrix to the second point cloud, if it exists to get them on same coordinate system         
            # if len(components) >= 20:  # Check if all transformation components are present
            #     transformation_matrix = np.array(components[4:20], dtype=float)  # Read all 16 components
            #     transformation_matrix = transformation_matrix.reshape(4, 4)  # Reshape to a 4x4 matrix
            #     # print(transformation_matrix)
            #     pcd_1 = pcd_1.transform(transformation_matrix)
            # else:
            #     transformation_matrix = None

            # Apply the ground truth transformation matrix to the second point cloud, if it exists to get them on same coordinate system         
            if len(components) >= 21:  # Adjusted the check here to 21 because of the added total_points
                transformation_matrix = np.array(components[5:21], dtype=float)  # Adjusted the start index here to 5
                transformation_matrix = transformation_matrix.reshape(4, 4)  # Reshape to a 4x4 matrix
                pcd_1 = pcd_1.transform(transformation_matrix)
            else:
                transformation_matrix = None

            # Print the id and total points to see which experiment is running
            print(f"id: {id}, total points: {total_points}")

            # generate new "ground truth offset"
            ground_truth_transform = random_transformation_matrix(int(id), range_t, range_r)
            # print(ground_truth_transform)

            # apply to target only
            pcd_1 = pcd_1.transform(ground_truth_transform)

            # Run the experiment
            metrics, estimated_transform = run_single_experiment(algorithm, pcd_0, pcd_1, ground_truth_transform, voxelsize)
            # print(estimated_transform)

            # Add the overlap percentage to the metrics
            metrics['overlap'] = float(overlap)
            # Add the total points to the metrics
            metrics['total_points'] = int(total_points)
            print(metrics)

            # Save the estimated transformation matrix
            # Consider adding the filenames here?
            save_transformations(estimated_transform, ground_truth_transform, algorithm.__name__, dataset_name, id)

            # Append the metrics to the list
            # Consider adding the filenames here?
            full_metrics_list.append(metrics)

    # Save the evaluation metrics
    save_full_eval_metrics(full_metrics_list, metrics_file_path)

    print("Experiments complete.")


def run_single_experiment(algorithm, 
                          pcd0: o3d.geometry.PointCloud, 
                          pcd1: o3d.geometry.PointCloud, 
                          ground_truth_transform: np.ndarray, 
                          voxelsize: float):
    """
    Function to run a single registration experiment, i.e., register two point clouds using a specified algorithm.

    Parameters:
    algorithm: The registration algorithm.
    pcd0: The first point cloud.
    pcd1: The second point cloud.
    ground_truth_transform: The ground truth transformation matrix.
    voxelsize: Size of voxel to be used in the point cloud.

    Returns:
    metrics, estimated_transform: The evaluation metrics and estimated transformation matrix.
    """

    # Error handling in case of incorrect input types
    if not isinstance(pcd0, o3d.geometry.PointCloud) or not isinstance(pcd1, o3d.geometry.PointCloud):
        raise ValueError("Input point clouds should be of type o3d.geometry.PointCloud")

    # Run the algorithm
    runtime, estimated_transform, downsampled_points = run_algorithm(algorithm, pcd0, pcd1, voxelsize)

    # visualizations for checking!
    # draw_registration_result(pcd0, pcd1, np.identity(4), 'input point clouds')
    # draw_registration_result(pcd0, pcd1, ground_truth_transform, 'with ground truth')
    # draw_registration_result(pcd0, pcd1, estimated_transform, 'with estimated transform')

    # Make sure both transforms are numpy arrays
    assert isinstance(estimated_transform, np.ndarray), f"Expected estimated_transform to be a numpy array, got {type(estimated_transform)}"
    assert isinstance(ground_truth_transform, np.ndarray), f"Expected ground_truth_transform to be a numpy array, got {type(ground_truth_transform)}"

    # gets metrics as a dictionary
    metrics = get_full_evaluation_metrics(pcd0, pcd1, ground_truth_transform, estimated_transform, runtime, downsampled_points)

    return metrics, estimated_transform


def run_algorithm(algorithm: Callable[[o3d.geometry.PointCloud, o3d.geometry.PointCloud], np.ndarray],
                  source: o3d.geometry.PointCloud, 
                  target: o3d.geometry.PointCloud, 
                  voxelsize: float):
    """
    Function to run the registration algorithm and measure its runtime.

    Parameters:
    - algorithm: The registration algorithm.
    - source: The source point cloud.
    - target: The target point cloud.

    Returns:
    - runtime: The runtime of the algorithm.
    - estimated_transform: The transformation matrix estimated by the algorithm.
    """

    # use voxelsize here if desired
    # Move the downsampling here? could then move downsampled_points to here also
    # this is to overide the voxelsize input so it is handled by algorithm execution
    if algorithm.__name__ == "multi_scale_ICP":
        source  = downsample_point_cloud(source, 0.01)
        target = downsample_point_cloud(target, 0.01)
    else:
        source  = downsample_point_cloud(source, voxelsize)
        target = downsample_point_cloud(target, voxelsize)

    downsampled_points = calculate_total_points(source, target)

    start = time.time()  # Start time
    registration_result = algorithm(source, target)  # Running the algorithm
    end = time.time()  # End time
    runtime = end - start  # Calculating the runtime

    # print(registration_result)

    return runtime, registration_result, downsampled_points
