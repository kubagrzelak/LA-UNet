import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from .UNet import Unet
from .metrics import normalized_cross_correlation, calcMetrics

# Load testing datasets
print('[INFO] Load data...')
data = np.load('../data-arrays/test.npy')
print('[INFO] Load labels...')
labels = np.load('../data-arrays/test_labels.npy')

tensor_data = torch.Tensor(data).cuda()
tensor_labels = torch.Tensor(labels).cuda()
print(f'     image tensor: {tensor_data.shape}, {type(tensor_data)}')
print(f'     label tensor: {tensor_labels.shape}, {type(tensor_labels)}')

dataloader = DataLoader(TensorDataset(tensor_data, tensor_labels), batch_size=88)

data_size = len(dataloader)
print(f'     Batches: {data_size}')

# Initialize a model
print('[INFO] Initialize model...')
model = torch.load('../results/model.pth')
#model = torch.load('/content/drive/My Drive/KURF_Colab/model.pth', map_location=torch.device('cpu'))
sig = nn.Sigmoid()

ncc_history = torch.zeros(data_size)
dice_history = torch.zeros(data_size)
accuracy_history = torch.zeros(data_size)

print('[INFO] Testing...')
with torch.no_grad():
    for sample, (X,y) in enumerate(dataloader):
        print(f'     sample: {sample + 1}/{data_size}')
        pred = model(X)
        pred = sig(pred)
        pred = torch.round(pred)
        ncc_history[sample] = normalized_cross_correlation(pred, y)
        accuracy_history[sample], dice_history[sample] = calcMetrics(pred, y)

avg_ncc = ncc_history.mean()
print(f'Avg NCC: {avg_ncc}')
avg_accuracy = accuracy_history.mean()
print(f'Avg Accuracy: {avg_accuracy}')
avg_dice = dice_history.mean()
print(f'Avg Dice: {avg_dice}')