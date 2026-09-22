"""Loss functions for student segmentation training with soft pseudo-labels."""

import torch
import torch.nn.functional as F


def soft_bce_loss(logits, soft_targets):
    """BCE loss with soft targets from ensemble confidence maps.

    Args:
        logits: (B, 1, H, W) raw logits from the student model
        soft_targets: (B, 1, H, W) float in [0, 1] (ensemble vote fraction)

    Returns:
        Scalar loss
    """
    return F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="mean")


def dice_loss(logits, targets, smooth=1.0):
    """Differentiable Dice loss for boundary quality.

    Args:
        logits: (B, 1, H, W) raw logits
        targets: (B, 1, H, W) float in [0, 1]
        smooth: smoothing constant to avoid division by zero

    Returns:
        Scalar loss (1 - Dice)
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary Focal Loss — downweights easy examples, focuses on boundaries.

    Args:
        logits: (B, 1, H, W) raw logits
        targets: (B, 1, H, W) float in [0, 1]
        alpha: weighting factor for positive class
        gamma: focusing parameter (higher = more focus on hard examples)

    Returns:
        Scalar loss
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def combined_loss(logits, soft_targets, bce_weight=1.0, dice_weight=0.5,
                  use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
    """Combined loss: (BCE or Focal) + Dice.

    Args:
        logits: (B, 1, H, W) raw logits
        soft_targets: (B, 1, H, W) float in [0, 1]
        bce_weight: weight for BCE/Focal term
        dice_weight: weight for Dice term
        use_focal: if True, replace BCE with Focal Loss
        focal_alpha: alpha for focal loss
        focal_gamma: gamma for focal loss

    Returns:
        total_loss, bce_loss_val, dice_loss_val
    """
    if use_focal:
        bce = focal_loss(logits, soft_targets, alpha=focal_alpha, gamma=focal_gamma)
    else:
        bce = soft_bce_loss(logits, soft_targets)
    dice = dice_loss(logits, soft_targets)
    total = bce_weight * bce + dice_weight * dice
    return total, bce, dice
