import torch
import torch.nn.functional as F

def advanced_jump_penalty(logits, edge_index, cluster_labels, depth, gamma=0.5, boundary_weight=0.3):
    probs = F.softmax(logits, dim=1)
    i, j = edge_index
    pred_diff = torch.norm(probs[i] - probs[j], dim=1)
    depth_diff = torch.abs(depth[i] - depth[j])
    cluster_same = (cluster_labels[i] == cluster_labels[j])
    is_boundary = (depth_diff > 3.0) & (~cluster_same)


    penalty_cluster = pred_diff[cluster_same].mean() if cluster_same.any() else 0.0
    penalty_depth = pred_diff[depth_diff < 3.0].mean() if (depth_diff < 3.0).any() else 0.0
    dynamic_threshold = torch.quantile(pred_diff, 0.9)
    penalty_jump = pred_diff[pred_diff > dynamic_threshold].mean() if (pred_diff > dynamic_threshold).any() else 0.0
    penalty_boundary = (pred_diff * is_boundary.float()).mean() if is_boundary.any() else 0.0

    return penalty_cluster + penalty_depth + gamma * penalty_jump + boundary_weight * penalty_boundary
