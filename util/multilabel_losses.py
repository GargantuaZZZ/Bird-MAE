import torch
import torch.nn.functional as F
from torch import nn


class FocalLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        reduction="mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = bce * torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEWithRankingLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        ranking_weight=0.1,
        margin=1.0,
        reduction="mean",
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)
        ranking = self._pairwise_ranking_loss(logits, targets)
        return self.bce_weight * bce + self.ranking_weight * ranking

    def _pairwise_ranking_loss(self, logits, targets):
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return logits.new_tensor(0.0)

        logits = logits[valid]
        pos_mask = pos_mask[valid]
        neg_mask = neg_mask[valid]

        pos_logits = logits.unsqueeze(2)
        neg_logits = logits.unsqueeze(1)
        pair_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        pair_losses = F.relu(self.margin - pos_logits + neg_logits)
        pair_losses = pair_losses[pair_mask]

        if pair_losses.numel() == 0:
            return logits.new_tensor(0.0)
        if self.reduction == "sum":
            return pair_losses.sum()
        return pair_losses.mean()
