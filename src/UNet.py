import torch
from torch import nn

class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    
    Modified 0. Ronneberg, P. Fisher, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical Image Computing
    and Computer-Assisted Intervention, pages 234-241.
    Springer, 2015.
    """
    
    def __init__(self, in_ch, out_ch, ch = 32, num_layers = 5, drop_prob = 0.0):
        """
        Parameters
        ----------
        in_ch : int
            Number of channels (feature maps) in the input MRI to the U-Net model.
        out_ch : int
            Number of channels (feature maps) in the output segmented MRI to the U-Net model.
        ch : int, optional
            Number of output channels (feature maps) of the first convolution layer. The default is 32.
        num_layers : int, optional
            Number of down-sampling and up-sampling layers. Depth of the U-Net. The default is 5.
        drop_prob : float, optional
            Dropout probability. The default is 0.0.

        Returns
        -------
        None.
        """
        
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ch = ch
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        
        # build down-sampling
        self.down_sampling_layers = nn.ModuleList([ConvBlock(in_ch, ch, drop_prob)])
        for _ in range(num_layers - 1):
            self.down_sampling_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        
        # build flat-sampling
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        
        # build up-sampling
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
            
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_ch, kernel_size=1, padding=0, stride=1, bias=True),
            )
        )
        
    def forward(self, image):
        """
        Pass image through U-Net.

        Parameters
        ----------
        image : Tensor
            Input 4D Tensor of shape `(N, in_ch, H, W)`.

        Returns
        -------
        output : Tensor0
            Output 4D Tensor of shape `(N, out_ch, H, W)`.

        """
        skip_connection = []
        output = image
        
        # apply down-sampling layers
        for layer in self.down_sampling_layers:
            output = layer(output)
            skip_connection.append(output)
            output = nn.functional.max_pool2d(output, kernel_size=2, stride=2, padding=0)
        
        # apply flat-sampling layer
        output = self.conv(output)
        
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            copy_layer = skip_connection.pop()
            output = transpose_conv(output)
            
            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != copy_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != copy_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = nn.functional.pad(output, padding, "reflect")
            
            output = torch.cat([output, copy_layer], dim=1)
            output = conv(output)
        
        return output
            

class ConvBlock(nn.Module):
    """
    A Convolution Block that consists of two sub-blocks each built of 
    a convolution layer, instance normalization, Exponential Linear (ELU) and dropout.
    """
    
    def __init__(self, in_ch, out_ch, drop_prob):
        """
        Parameters
        ----------
        in_ch : int
            Number of channels (feature maps) in the input to the block.
        out_ch : int
            Number of channels (feature maps) in the output to the block.
        drop_prob : float
            Dropout probability.

        Returns
        -------
        None.
        """
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.drop_prob = drop_prob
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, stride=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(alpha=0.1, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2, stride=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(alpha=0.1, inplace=True),
            nn.Dropout2d(drop_prob),
        )
        
    def forward(self, image):
        """
        Pass image through this convolution block.

        Parameters
        ----------
        mri : Tensor
            Input 4D tensor of shape `(N, in_ch, H, W)`.

        Returns
        -------
        Tensor
            Output 4D tensor of shape `(N, out_ch, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block built of
    a convolution transpose layer, instance normalization, Exponential Linear (ELU)
    """
    
    def __init__(self, in_ch, out_ch):
        """
        Parameters
        ----------
        in_ch : int
            Number of channels (feature maps) in the input to the block.
        out_ch : int
            Number of channels (feature maps) in the output to the block.

        Returns
        -------
        None.
        """
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(alpha=0.1, inplace=True),
        )
    
    def forward(self, image):
        """
        Pass image through this transpose convolution block.

        Parameters
        ----------
        image : Tensor
            Input 4D tensor of shape `(N, in_ch, H, W)`.

        Returns
        -------
        Tensor
            Output 4D tensor of shape `(N, out_ch, H*2, W*2)`.

        """
        return self.layers(image)
