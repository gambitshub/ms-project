import numpy as np
import copy
import math

success_rte_thresh = 0.6  # translation threshold
success_rre_thresh = 30  # rotation threshold


def rte_rre(T_pred, T_gt, rte_thresh=success_rte_thresh, rre_thresh=success_rre_thresh, eps=1e-16):
    if T_pred is None:
        return False, np.inf, np.inf

    rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
    rre = np.arccos(
        np.clip((np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps,
                1 - eps)) * 180 / math.pi

    return rte < rte_thresh and rre < rre_thresh, rte, rre


def rmse(array1, array2):
    return np.sqrt(np.mean((array1 - array2) ** 2))


def get_rsme(pcd, T_pred, T_gt):
    pcd_est = copy.deepcopy(pcd)
    pcd_est.transform(T_pred)
    pcd_true = copy.deepcopy(pcd)
    pcd_true.transform(T_gt)

    pcd_est = np.asarray(pcd_est.points)
    pcd_true = np.asarray(pcd_true.points)

    return rmse(pcd_est, pcd_true)


def get_averages(metrics):
    metrics = np.array(metrics, dtype=object)

    avg_time = np.mean(metrics[:, 0])
    avg_rsme = np.mean(metrics[:, 1])
    avg_rte = np.mean(metrics[:, 2])
    avg_rre = np.mean(metrics[:, 3])

    successful_metrics = metrics[metrics[:, 4] == 1]
    s_avg_time = np.mean(successful_metrics[:, 0])
    s_avg_rsme = np.mean(successful_metrics[:, 1])
    s_avg_rte = np.mean(successful_metrics[:, 2])
    s_avg_rre = np.mean(successful_metrics[:, 3])

    recall = len(successful_metrics) / len(metrics)

    return avg_time, avg_rsme, avg_rte, avg_rre, s_avg_time, s_avg_rsme, s_avg_rte, s_avg_rre, recall

