import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        """
        多标签 Focal Loss

        gamma: 调节因子，越大越关注困难样本
        alpha: 类别平衡因子，控制正负样本权重
        reduction: mean / sum / none
        """
        super(MultiLabelFocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_samplewise_focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    返回每个样本的 Focal Loss 总和（shape: [B]）
    """
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    return focal_loss
