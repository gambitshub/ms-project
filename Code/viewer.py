"""
viewer.py

Author: James Daniels

"""
from crosssource_loader import CrosssourceDataset
from pathlib import Path
from utils import make_open3d_point_cloud
import open3d as o3d
import os
import numpy as np
import copy
import argparse

def load_point_clouds():
    pc1_path = f"data/cross-source-dataset/kinect_lidar/scene5/pair1/kinect.ply"
    pc_1 = o3d.io.read_point_cloud(pc1_path)

    pc2_path = f"data/cross-source-dataset/kinect_lidar/scene5/pair1/lidar.ply"
    pc_2 = o3d.io.read_point_cloud(pc2_path)

    return pc_1, pc_2

def load_transformations(algorithm, dataset, idx):
    trans_path = f"{algorithm}_results/{dataset}/results/{idx}.txt"
    
    with open(trans_path, 'r') as f:
        lines = f.readlines()

    est_lines = lines[1:5]
    gt_lines = lines[7:11]

    est_trans = np.array([list(map(float, line.strip().split())) for line in est_lines])
    gt_trans = np.array([list(map(float, line.strip().split())) for line in gt_lines])

    return est_trans, gt_trans

def draw_registration_result(source, target, transformation,name="Result"):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],width=1200,height=800,left=400,top=150,window_name=name)


def main(args):
    # algorithm = args.algorithm
    # dataset = args.dataset
    # idx = args.idx

    pc_1, pc_2 = load_point_clouds()

    # est_trans, gt_trans = load_transformations(algorithm, dataset, idx)

    print("Loaded point clouds:")
    print(pc_1)
    print(pc_2)
    # print("Estimated transformation:", est_trans)
    # print("Ground truth transformation:", gt_trans)

    source = pc_1
    target = pc_2

    draw_registration_result(source, target, np.identity(4), "Input point clouds")
    # draw_registration_result(source, target, est_trans, "Point cloud transformation estimate applied")
    # draw_registration_result(source, target, gt_trans, "Point cloud with ground truth applied")

    # if algorithm == "FGR":
    #     print("here")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize point cloud registration.')
    # parser.add_argument('algorithm', type=str, help='Algorithm (e.g., FGR)')
    # parser.add_argument('dataset', type=str, help='Dataset directory (e.g., point_clouds)')
    # parser.add_argument('idx', type=int, help='ID# (e.g., 1)')
    args = parser.parse_args()
    main(args)

# if __name__ == "__main__":
#     algorithm = input("Enter the algorithm (e.g., FGR): ")
#     dataset = input("Enter the dataset directory (e.g., point_clouds): ")
#     idx = int(input("Enter the id# (e.g., 1): "))
#     main(algorithm, dataset, idx)