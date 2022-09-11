import torch
from torch import nn

class DiceLoss(nn.Module):
    """
    Implementation of Dice Loss.
    The Dice loss ranges between
    0: perfectly matching the ground truth distribution  
    1: complete mismatch with the ground truth.
    """
    
    def __init__(self):
        """
        Returns
        -------
        None.
        """
        super(DiceLoss, self).__init__()
    
    def forward(self, y_hat, y, axis=(1, 2, 3), epsilon=0.00001):
        """
        Calculate soft dice loss for predicted segmentation and ground truth.

        Parameters
        ----------
        y_hat : Tensor
            Predicted segmentation as 4D tensor of shape `(N, in_ch, H, W)`.
        y : Tensor
            Ground truth segmentation as 4D tensor of shape `(N, in_ch, H, W)`.
        axis: Tuple
            Spatial axes to sum over when computing numerator and denominator in formula for dice loss.
        epsilon: Float
            Small constant for mathematical soundness and to avoid dividing by zero.

        Returns
        -------
        Tensor
            Scalar.

        """
        numerator_1 = torch.sum(y_hat * y, axis=axis) + epsilon
        denominator_1 = torch.sum(y_hat + y, axis=axis) + epsilon
        numerator_2 = torch.sum((1 - y_hat) * (1 - y), axis=axis) + epsilon
        denominator_2 = torch.sum(2 - y_hat - y, axis=axis) + epsilon
        dice_loss = 1 - torch.mean(numerator_1 / denominator_1) - torch.mean(numerator_2 / denominator_2)
        return dice_loss

class SoftDiceLoss(nn.Module):
    """
    Implementation of Soft Dice Loss that accept continuous probabilities for predictions.
    The soft Dice loss ranges between
    0: perfectly matching the ground truth distribution  
    1: complete mismatch with the ground truth.
    """
    
    def __init__(self):
        """
        Returns
        -------
        None.
        """
        super(SoftDiceLoss, self).__init__()
    
    def forward(self, y_hat, y, axis=(1, 2, 3), epsilon=0.00001):
        """
        Calculate soft dice loss for predicted segmentation and ground truth.

        Parameters
        ----------
        y_hat : Tensor
            Predicted segmentation as 4D tensor of shape `(N, in_ch, H, W)`.
        y : Tensor
            Ground truth segmentation as 4D tensor of shape `(N, in_ch, H, W)`.
        axis: Tuple
            Spatial axes to sum over when computing numerator and denominator in formula for dice loss.
        epsilon: Float
            Small constant for mathematical soundness and to avoid dividing by zero.

        Returns
        -------
        Tensor
            Scalar.

        """
        dice_numerator = 2 * torch.sum(y_hat * y, axis=axis) + epsilon
        dice_denominator = torch.sum(y_hat ** 2, axis=axis) + torch.sum(y ** 2, axis=axis) + epsilon
        dice_loss = 1 - torch.mean(dice_numerator / dice_denominator)
        return dice_loss