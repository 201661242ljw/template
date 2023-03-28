import torch
import torch.nn as nn


def MyLoss_d(predictions, targets):
    """
    计算描述3个点距离之和的损失
    Args:
        predictions: 预测值, shape=(batch_size, 6)
        targets: 真实值, shape=(batch_size, 6)
    Returns:
        distance_loss: 描述3个点距离之和的损失
    """
    # 将预测值和真实值分别按照 x 和 y 的坐标分开，reshape 成 (batch_size, 3, 2)

    pred_points = torch.reshape(predictions, (-1, 3, 2))
    target_points = torch.reshape(targets, (-1, 3, 2))

    # 计算每对点之间的距离
    distances = torch.sqrt(torch.sum(torch.square(pred_points - target_points), axis=-1))

    # 计算描述3个点距离之和的损失
    distance_loss = torch.mean(torch.sum(distances, axis=-1))
    return distance_loss


def MyLoss_heatmap(output, target, target_weight):
    batch_size = output.size(0)
    num_joints = output.size(1)

    target_weight = target_weight.reshape((batch_size, num_joints, 1, 1))

    output = output * target_weight
    target = target * target_weight
    heatmaps_pred = output.reshape((batch_size, num_joints, -1))
    heatmaps_gt = target.reshape((batch_size, num_joints, -1))

    loss = 0.0
    # loss += nn.MSELoss(heatmaps_gt, heatmaps_pred)
    mse_loss = nn.MSELoss()
    loss += mse_loss(heatmaps_gt, heatmaps_pred)
    return loss / num_joints
