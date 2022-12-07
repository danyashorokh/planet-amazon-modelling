import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from torchvision.ops.focal_loss import sigmoid_focal_loss


class BCEWithLogitsLossSmoothing(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logit, target):
        target = target.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logit, target, reduction=self.reduction)


class FocalLoss(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logit, target):
        target = target.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return sigmoid_focal_loss(logit, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
