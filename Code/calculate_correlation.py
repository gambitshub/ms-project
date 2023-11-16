from scipy.spatial.transform import Rotation as R
from pathlib import Path
from utils import make_open3d_point_cloud
import open3d as o3d
import os
import numpy as np
import copy
import argparse
import pandas as pd
import gc
import json
from scipy.stats import spearmanr

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize point cloud registration.')
    parser.add_argument('-dataset', type=str, help='Dataset name')
    parser.add_argument('-algorithms', nargs='+', type=str, 
                        default=["RANSAC", "ICP", "multi_scale_ICP", "FGR", "point_to_plane_ICP", "FMR"],
                        help='Algorithm names')
    parser.add_argument('-id', type=int, default = 0, help='ID# (e.g., 1)')
    return parser.parse_args()

def get_evaluation_metrics(dataset_name: str, algorithm: str, index: int):

    directory = Path(".") / "results" / dataset_name / algorithm

    metrics_file_path = directory / "voxelsize0.02_rangeT0.5_rangeR60_overlap0.5_metrics.txt"

    with open(metrics_file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == index:
                metrics = json.loads(line)
                return metrics
            

# def get_initial_misalignment(transformation_matrix):
#     # Extract translation (last column except the last entry) from the transformation matrix
#     # print(transformation_matrix)
#     translation = transformation_matrix[:3, 3]

#     # Compute Euclidean distance as a measure of misalignment
#     translational_misalignment = np.linalg.norm(translation)

#     # Extract the rotation component of the transformation matrix
#     rotation = transformation_matrix[:3, :3]

#     # Convert the rotation matrix to an angle-axis representation
#     rot = R.from_matrix(rotation)
#     angle = rot.as_rotvec()

#     # Compute the rotational misalignment as the magnitude of the rotation vector
#     rotational_misalignment = np.linalg.norm(angle)

#     # Combine the translational and rotational misalignment by adding them
#     total_misalignment = translational_misalignment + rotational_misalignment

#     return total_misalignment

def get_initial_misalignment(transformation_matrix):
    # Identity matrix for comparison
    identity_matrix = np.eye(4)

    # Compute the Frobenius norm as a measure of misalignment
    frobenius_norm = np.linalg.norm(transformation_matrix - identity_matrix, 'fro')

    return frobenius_norm


def load_transformations(algorithm, dataset, idx):
    trans_path = f"./results/{dataset}/{algorithm}/transformations/{idx}.txt"
    with open(trans_path, 'r') as f:
        lines = f.readlines()
    est_trans = []
    gt_trans = []
    current_trans = est_trans
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] in ['Estimated', 'Ground']:
            if parts[0] == 'Ground':
                current_trans = gt_trans
            continue
        current_trans.append([float(part) for part in parts])
    est_trans = np.array(est_trans)
    gt_trans = np.array(gt_trans)
    if est_trans.shape != (4, 4) or gt_trans.shape != (4, 4):
        raise ValueError(f"Invalid transformation matrices in {trans_path}")
    return est_trans, gt_trans

def collect_data(dataset, algorithms, indices):
    data = []
    for idx in indices:
        for algorithm in algorithms:
            metrics = get_evaluation_metrics(dataset, algorithm, idx)
            est_trans, gt_trans= load_transformations(algorithm, dataset, idx)
            initial_misalignment = get_initial_misalignment(gt_trans)
            # print(initial_misalignment)
            data.append({
                'algorithm': algorithm,
                'problem_id': idx,
                'initial_misalignment': initial_misalignment,
                'combined_error_metric': metrics['combined_error_metric'],
                'overlap': metrics['overlap']
            })
    return pd.DataFrame(data)


def calculate_correlations(data):
    algorithms = data['algorithm'].unique()
    correlations = {}
    for algorithm in algorithms:
        subset = data[data['algorithm'] == algorithm]
        correlations[algorithm] = {
            'initial_misalignment_vs_combined_error_metric': spearmanr(subset['initial_misalignment'], subset['combined_error_metric']),
            'overlap_vs_combined_error_metric': spearmanr(subset['overlap'], subset['combined_error_metric'])
        }
        # correlations[algorithm] = {
        #     'initial_misalignment_vs_combined_error_metric': subset['initial_misalignment'].corr(subset['combined_error_metric']),
        #     'overlap_vs_combined_error_metric': subset['overlap'].corr(subset['combined_error_metric'])
        # }
    return correlations


def main():
    args = parse_args()
    dataset = args.dataset
    indices = range(182)  # Adjust as needed

    data = collect_data(dataset, args.algorithms, indices)
    correlations = calculate_correlations(data)
    for algorithm, corr in correlations.items():
        print(f"Algorithm: {algorithm}")
        print(f"Correlations: {corr}")
        print()

if __name__ == "__main__":
    main()
