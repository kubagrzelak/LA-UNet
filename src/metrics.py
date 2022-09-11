import torch
import numpy as np

def normalized_cross_correlation(x, y, eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
  
    ncc = torch.mean(torch.sum(ncc, dim=1))
    
    return ncc

def calcMetrics(A, B):  # A is predicted, B is ground truth
        A = A.cpu().numpy()
        B = B.cpu().numpy()

        A = A.astype(bool)
        B = B.astype(bool)

        TP = np.sum(np.logical_and(A, B))
        FP = np.sum(np.logical_and(A, np.logical_not(B)))
        TN = np.sum(np.logical_and(np.logical_not(A), np.logical_not(B)))
        FN = np.sum(np.logical_and(np.logical_not(A), B))

        dice = 2 * TP / (2 * TP + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        return accuracy, dice

# def calcMetrics(A, B):  # A is predicted, B is ground truth
#     metrics = {}

#     A = A.cpu().numpy()
#     B = B.cpu().numpy()

#     A = A.astype(bool)
#     B = B.astype(bool)

#     TP = np.sum(np.logical_and(A, B))
#     FP = np.sum(np.logical_and(A, np.logical_not(B)))
#     TN = np.sum(np.logical_and(np.logical_not(A), np.logical_not(B)))
#     FN = np.sum(np.logical_and(np.logical_not(A), B))

#     metrics['dice'] = 2 * TP / (2 * TP + FP + FN)
#     metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
#     metrics['sensitivity'] = TP / (TP + FN)
#     metrics['specificity'] = TN / (TN + FP)
#     metrics['precision'] = TP / (TP + FP) if TP + FP != 0 else 0
#     V_A = np.sum(A)
#     V_B = np.sum(B)
#     metrics['volume_diff'] = abs(V_A - V_B)
#     metrics['volume_diff_percentage_error'] = (abs(V_A - V_B)/V_B) * 100
    
#     return metrics