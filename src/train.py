import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from .UNet import Unet
from .metrics import normalized_cross_correlation

torch.cuda.empty_cache()

# Load training datasets
print('[INFO] Load training data...')
training_data = np.load('../data-arrays/training.npy')
print('[INFO] Load training labels...')
training_labels = np.load('../data-arrays/training_labels.npy')

tensor_data = torch.Tensor(training_data).cuda()
tensor_labels = torch.Tensor(training_labels).cuda()
print(f'     image tensor: {tensor_data.shape}, {type(tensor_data)}')
print(f'     label tensor: {tensor_labels.shape}, {type(tensor_labels)}')

training_dataloader = DataLoader(TensorDataset(tensor_data, tensor_labels), shuffle=True, batch_size=16)

data_size = len(training_dataloader)
print(f'     Batches: {data_size}')

# Initialize a model
print('[INFO] Initialize model...')
# Number of channels (feature maps) in the input MRI to the U-Net model.
in_ch = 1
# Number of channels (feature maps) in the output segmented MRI to the U-Net model.
out_ch = 1
# Number of output channels (feature maps) of the first convolution layer. The default is 32.
ch = 32
# Number of down-sampling and up-sampling layers. Depth of the U-Net. The default is 5.
num_layers = 5
# Dropout probability. The default is 0.0.
drop_prob = 0.0
model = Unet(in_ch=in_ch, out_ch=out_ch, ch=ch, num_layers=num_layers, drop_prob=drop_prob)
model.cuda()

# Hyperparameters
learning_rate = 0.001
epochs = 20

# Loss Function
print('[INFO] Initialize loss function...')
loss_fn = nn.BCEWithLogitsLoss()

# History of loss function values
loss_avg_epochs = torch.zeros(epochs)
ncc_avg_epochs = torch.zeros(epochs)

# Optimizer
print('[INFO] Initialize optimizer ADAM...')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training
print('[INFO] Start training...')
for e in range(epochs):
    print(f"\n--------------- Epoch {e+1} ---------------")
    loss_history_epoch = torch.zeros(data_size)
    ncc_history_epoch = torch.zeros(data_size)

    for batch, (X,y) in enumerate(training_dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_history_epoch[batch] = loss.item()
        ncc_history_epoch[batch] = normalized_cross_correlation(pred,y)
        
        # Backpropagation
        optimizer.zero_grad() # reset grads to avoid double counting
        loss.backward()
        optimizer.step()    # adjust parameters
        
        if batch == 0 or (batch + 1) % 55 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch:>5d}/{data_size:>5d}]")
    
    loss_avg_epochs[e] = float(loss_history_epoch.mean())
    print(f"EPOCH {e+1} LOSS AVG:    {loss_avg_epochs[e]}")
    ncc_avg_epochs[e] = float(ncc_history_epoch.mean())
    print(f"EPOCH {e+1} NCC AVG:    {ncc_avg_epochs[e]}")

print('[INFO] Saving loss history and model...')
torch.save(loss_avg_epochs, '../results/loss_avg_epochs.pt')
torch.save(ncc_avg_epochs, '../results/ncc_avg_epochs.pt')
torch.save(model, '../results/model.pth')

print('[INFO] Plotting training loss...')
plt.plot(loss_avg_epochs)
plt.ylabel('Average Loss per Epoch (BCEWithLogitsLoss)')
plt.xlabel('Epoch')
plt.show()

print('[INFO] Plotting training ncc...')
plt.plot(ncc_avg_epochs)
plt.ylabel('Average NCC per Epoch')
plt.xlabel('Epoch')
plt.show()