"""
Demo the feature-metric registration algorithm
Creator: Xiaoshui Huang
Date: 2021-04-13
"""
import os
import sys
import copy
import open3d
import torch
import torch.utils.data
import logging
import numpy as np
from model import PointNet, Decoder, SolveRegistration
import se_math.transforms as transforms

from evaluation import get_rsme
from evaluation import rte_rre
from evaluation import get_averages
from crosssource_loader import CrosssourceDataset
import time

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# def draw_registration_result(source: np.ndarray, target: np.ndarray, transformation: np.ndarray, name: str = "Result") -> None:
def draw_registration_result(source, target, transformation, name="Result"):
    """
    Function to visualize the result of registration.
    Draws the registration results between source and target point clouds using Open3D.
    Draws the registration result of source and target point clouds after applying the transformation.
    
    Parameters:
    - source: The source point cloud.
    - target: The target point cloud.
    - transformation: The transformation matrix.
    - name: The window name.
    #     Parameters:
#     source (np.ndarray): The source point cloud data.
#     target (np.ndarray): The target point cloud data.
#     transformation (np.ndarray): The transformation matrix to apply on the source point cloud.
#     name (str): The window name for the visualization. Defaults to "Result".
    """
    # Deepcopy is used to prevent any changes to the original data
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # Painting each point cloud with a different color for visual distinction
    source_temp.paint_uniform_color([1, 0.706, 0])  # Gold color for source
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue color for target

    # Applying the estimated transformation to the source point cloud
    source_temp.transform(transformation)
    
    # Visualizing the transformed source and original target point clouds
    o3d.visualization.draw_geometries([source_temp, target_temp],width=1200,height=800,left=400,top=150,window_name=name)


class Demo:
    def __init__(self):
        self.dim_k = 1024
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 1  # see. self.compute_loss()

    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points, not used during the evaluation
        decoder = Decoder()
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder, isTest=True)
        return fmr_solver

    def evaluate(self, solver, p0, p1, device):
        solver.eval()
        with torch.no_grad():
            p0 = torch.tensor(p0,dtype=torch.float).to(device)  # template (1, N, 3)
            p1 = torch.tensor(p1,dtype=torch.float).to(device)  # source (1, M, 3)
            solver.estimate_t(p0, p1, self.max_iter)

            est_g = solver.g  # (1, 4, 4)
            g_hat = est_g.cpu().contiguous().view(4, 4)  # --> [1, 4, 4]

            return g_hat


def FMR(p1, p0):
    downp0 = p0.voxel_down_sample(voxel_size=0.1)
    newp0 = np.asarray(downp0.points)
    newp0 = np.expand_dims(newp0,0)
    downp1 = p1.voxel_down_sample(voxel_size=0.1)
    newp1 = np.asarray(downp1.points)
    newp1 = np.expand_dims(newp1,0)
    # print(p1)
    fmr = Demo()
    model = fmr.create_model()
    # pretrained_path = "./result/fmr_model_7scene.pth"
    # pretrained_path = "/mnt/c/Users/James/fp/feature_metric_reg/result/fmr_model_7scene.pth"
    pretrained_path = "/mnt/c/Users/James/fp/feature_metric_reg/result/fmr_model.pth"
    # pretrained_path = "./result/fmr_model_modelnet40.pth"
    # pretrained_path = "/mnt/c/Users/James/fp/feature_metric_reg/result/fmr_model_modelnet40.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    device = "cpu"
    model.to(device)

    T_est = fmr.evaluate(model, newp0, newp1, device)

    print(T_est)

    # draw_registration_result(p1_pcd, p0_pcd, T_est)

    # Clear memory
    downp0.clear()
    downp1.clear()
    torch.cuda.empty_cache()
    del model

    # Convert to numpy array
    if isinstance(T_est, torch.Tensor):
        registration_result = T_est.cpu().numpy()

    return registration_result