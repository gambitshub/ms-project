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

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize point cloud registration.')
    parser.add_argument('-dataset', type=str, help='Dataset name')
    parser.add_argument('-algorithms', nargs='+', type=str, 
                        default=["RANSAC", "ICP", "multi_scale_ICP", "FGR", "point_to_plane_ICP", "FMR"],
                        help='Algorithm names')
    parser.add_argument('-id', type=int, default = 0, help='ID# (e.g., 1)')
    return parser.parse_args()

# def get_evaluation_metrics(dataset_name: str, algorithm: str, index: int):
#     if dataset_name == "cross-source-dataset":
#         directory = Path(".") / "results/cross-source-dataset" / algorithm
#     elif dataset_name == "sun3d-hotel_umd-maryland_hotel3":
#         directory = Path(".") / "results/sun3d-hotel_umd-maryland_hotel3" / algorithm
#     elif dataset_name == "gazebo_winter":
#         directory = Path(".") / "results" / dataset_name / algorithm

#     metrics_file_path = directory / "voxelsize0.02_rangeT0.5_rangeR60_overlap0.5_metrics.txt"

#     with open(metrics_file_path, 'r') as file:
#         for i, line in enumerate(file):
#             if i == index:
#                 metrics = json.loads(line)
#                 return metrics
            
def get_evaluation_metrics(dataset_name: str, algorithm: str, index: int):

    directory = Path(".") / "results" / dataset_name / algorithm

    metrics_file_path = directory / "voxelsize0.02_rangeT0.5_rangeR60_overlap0.5_metrics.txt"

    with open(metrics_file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == index:
                metrics = json.loads(line)
                return metrics
            

def display_results_table(dataset, algorithms, idx):
    results = []
    for algorithm in algorithms:
        metrics = get_evaluation_metrics(dataset, algorithm, idx)
        results.append(metrics)

    # Create DataFrame using first result to get column names (metric names)
    df = pd.DataFrame([results[0]], columns=results[0].keys())

    # Add the rest of the results to DataFrame
    dataframes = [df]
    for i in range(1, len(results)):
        dataframes.append(pd.DataFrame([results[i]], columns=results[i].keys()))
    
    df = pd.concat(dataframes, ignore_index=True)

    df.insert(0, 'Algorithm', algorithms)
    df.set_index('Algorithm', inplace=True)
    # Allow pandas to display all columns of the DataFrame.
    pd.set_option('display.max_columns', None)
    print(df)

def get_filename_and_ground_truth(dataset_name: str, overlap_threshold: float, id: int):
    if dataset_name == "cross-source-dataset":
        directory = Path(".") / "data/cross-source-dataset"
    elif dataset_name in ["sun3d-hotel_umd-maryland_hotel3", "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]:
        directory = Path(".") / "data/SUN3D" / dataset_name
    elif dataset_name in ["gazebo_winter", "hauptgebaude"]:
        directory = Path(".") / "data/ETH" / dataset_name

    overlap_file_path = directory / f"overlapping_point_clouds_overlap_{overlap_threshold}.txt"
    data = pd.read_csv(overlap_file_path)
    row = data[data['id'] == id].iloc[0]
    filename1 = row['filename1']
    filename2 = row['filename2']
    overlap = row['overlap']
    total_points = row['total_points']
    if dataset_name == "gazebo_winter":
        transformation_matrix = np.identity(4)
    else:
        transformation_matrix = row[[f'transform_{i}' for i in range(16)]].values.reshape(4, 4)
    print(filename1)
    print(filename2)
    return (filename1, filename2), overlap, total_points, transformation_matrix


def load_point_clouds(dataset_name, filename1, filename2):
    if dataset_name == "cross-source-dataset":
        directory = Path(".") / "data/cross-source-dataset"
    elif dataset_name in ["sun3d-hotel_umd-maryland_hotel3", "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]:
        directory = Path(".") / "data/SUN3D" / dataset_name
    elif dataset_name in ["gazebo_winter", "hauptgebaude"]:
        directory = Path(".") / "data/ETH" / dataset_name

    if dataset_name == "gazebo_winter":
        point_clouds = []
        for filename in [filename1, filename2]:
            df = pd.read_csv(os.path.join(directory, filename))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
            point_clouds.append(pcd)

        return point_clouds[0], point_clouds[1]
    
    else:
        pc1_path = directory / f"{filename1}"
        pc_1 = o3d.io.read_point_cloud(str(pc1_path))
        points_1 = np.asarray(pc_1.points)
        pc_1 = make_open3d_point_cloud(points_1)

        pc2_path = directory / f"{filename2}"
        pc_2 = o3d.io.read_point_cloud(str(pc2_path))
        points_2 = np.asarray(pc_2.points)
        pc_2 = make_open3d_point_cloud(points_2)

        return pc_1, pc_2
    

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


def draw_registration_result(source, target, transformation,name="Result"):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],width=1200,height=800,left=400,top=150,window_name=name)


def main():
    args = parse_args()
    dataset = args.dataset
    idx = args.id

    (filename1, filename2), _, _, transformation_matrix = get_filename_and_ground_truth(dataset, 0.5, idx)
    # print(transformation_matrix)

    pcd_0, pcd_1 = load_point_clouds(dataset, filename1, filename2)
    pcd_1 = pcd_1.transform(transformation_matrix)

    first_algorithm = args.algorithms[0]
    _, gt_trans = load_transformations(first_algorithm, dataset, idx)
    source = pcd_0
    target = pcd_1.transform(gt_trans)

    draw_registration_result(source, target, np.identity(4), "Input point clouds to registration problem")
    draw_registration_result(source, target, gt_trans, "Point cloud with ground truth applied")


    for algorithm in args.algorithms:
        est_trans, _ = load_transformations(algorithm, dataset, idx)
        # print("Loaded point clouds:")
        # print(pcd_0)
        # print(pcd_1)

        print(f"Estimated transformation using {algorithm}:", est_trans)

        draw_registration_result(source, target, est_trans, f"Point cloud transformation estimate applied - {algorithm}")

        # display_evaluation_metrics(dataset, algorithm, idx)

    display_results_table(dataset, args.algorithms, idx)

    del pcd_0, pcd_1, source, target
    gc.collect()


if __name__ == "__main__":
    main()


# def display_evaluation_metrics(dataset_name: str, algorithm: str, index: int):
#     if dataset_name == "cross-source-dataset":
#         directory = Path(".") / "results/cross-source-dataset" / algorithm
#     elif dataset_name == "sun3d-hotel_umd-maryland_hotel3":
#         directory = Path(".") / "results/sun3d-hotel_umd-maryland_hotel3" / algorithm
#     elif dataset_name == "gazebo_winter":
#         directory = Path(".") / "results" / dataset_name / algorithm

#     metrics_file_path = directory / "voxelsize0.02_rangeT0.5_rangeR60_overlap0.5_metrics.txt"
    
#     with open(metrics_file_path, 'r') as file:
#         for i, line in enumerate(file):
#             if i == index:
#                 metrics = json.loads(line)
#                 print("Evaluation Metrics:")
#                 for key, value in metrics.items():
#                     print(f"{key}: {value}")
#                 break

# def main(args):
#     dataset = args.dataset
#     algorithm = args.algorithm
#     idx = args.idx

#     (filename1, filename2), _, _, transformation_matrix = get_filename_and_ground_truth(dataset, 0.5, idx)
#     print(transformation_matrix)

#     pcd_0, pcd_1 = load_point_clouds(dataset, filename1, filename2)
#     est_trans, gt_trans = load_transformations(algorithm, dataset, idx)

#     print("Loaded point clouds:")
#     print(pcd_0)
#     print(pcd_1)

#     pcd_1 = pcd_1.transform(transformation_matrix)

#     print("Estimated transformation:", est_trans)
#     print("Ground truth transformation:", gt_trans)

#     source = pcd_0
#     target = pcd_1
#     pcd_1 = pcd_1.transform(gt_trans)

#     draw_registration_result(source, target, np.identity(4), "Input point clouds to registration problem")
#     draw_registration_result(source, target, est_trans, "Point cloud transformation estimate applied")
#     draw_registration_result(source, target, gt_trans, "Point cloud with ground truth applied")

#     display_evaluation_metrics(dataset, algorithm, idx)

#     del pcd_0, pcd_1, est_trans, gt_trans, transformation_matrix, source, target
#     gc.collect()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Visualize point cloud registration.')
#     parser.add_argument('dataset', type=str, help='Dataset name')
#     parser.add_argument('algorithm', type=str, help='Algorithm name')
#     parser.add_argument('idx', type=int, help='ID# (e.g., 1)')
#     args = parser.parse_args()
#     main(args)




# from pathlib import Path
# from utils import make_open3d_point_cloud
# import open3d as o3d
# import os
# import numpy as np
# import copy
# import argparse
# import pandas as pd
# import gc

# # This extract the point cloud filenames and the respective transformations
# def get_filename_and_ground_truth(dataset_name: str, overlap_threshold: float, id: int):
#     if dataset_name == "cross-source-dataset":
#         directory = Path(".") / "data/cross-source-dataset"
#     elif dataset_name == "sun3d-hotel_umd-maryland_hotel3":
#         directory = Path(".") / "data/SUN3D/sun3d-hotel_umd-maryland_hotel3"

#     overlap_file_path = directory / f"overlapping_point_clouds_overlap_{overlap_threshold}.txt"
#     data = pd.read_csv(overlap_file_path)
#     row = data[data['id'] == id].iloc[0]
#     filename1 = row['filename1']
#     filename2 = row['filename2']
#     overlap = row['overlap']
#     total_points = row['total_points']
#     transformation_matrix = row[[f'transform_{i}' for i in range(16)]].values.reshape(4, 4)
#     print(filename1)
#     print(filename2)
#     return (filename1, filename2), overlap, total_points, transformation_matrix

# def make_open3d_point_cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
#     if isinstance(xyz, o3d.geometry.PointCloud):
#         return xyz
#     else:
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         return pcd

# def load_point_clouds(dataset_name, filename1, filename2):
#     if dataset_name == "cross-source-dataset":
#         directory = Path(".") / "data/cross-source-dataset"
#     elif dataset_name == "sun3d-hotel_umd-maryland_hotel3":
#         directory = Path(".") / "data/SUN3D/sun3d-hotel_umd-maryland_hotel3"

#     pc1_path = directory / f"{filename1}"
#     pc_1 = o3d.io.read_point_cloud(str(pc1_path))
#     points_1 = np.asarray(pc_1.points)
#     pc_1 = make_open3d_point_cloud(points_1)

#     pc2_path = directory / f"{filename2}"
#     pc_2 = o3d.io.read_point_cloud(str(pc2_path))
#     points_2 = np.asarray(pc_2.points)
#     pc_2 = make_open3d_point_cloud(points_2)

#     return pc_1, pc_2