import torch 
import torch.nn as nn
from torchvision import transforms

class UNetEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, crop_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.center_crop = transforms.CenterCrop(crop_size)

        self.relu = nn.ReLU()

        self.conv_layer_1 = nn.Conv2d(in_channels = self.in_channels,
                                      out_channels = self.out_channels,
                                      kernel_size = self.kernel_size)
        
        self.conv_layer_2 = nn.Conv2d(in_channels = self.out_channels,
                                      out_channels = self.out_channels,
                                      kernel_size = self.kernel_size)
        
        self.max_pool = nn.MaxPool2d(2, stride = 2)

    def forward(self, x):
        """
        Args:
        -  torch.tensor : x (B x C x H x W)

        Return:
        - torch.tensor : x (B x (out_channels)x ((H-2*kernel_size//2)//2 + 1) x ((W-2*kernel_size//2)//2 + 1))
        - torch.tensor : x_skip (B x (out_channels) x crop_size x crop_size)
        """

        x = self.relu(self.conv_layer_1(x)) # B x (out_channels) x (H-kernel_size-1) x (W-kernel_size-1)
        x = self.relu(self.conv_layer_2(x)) # B x (out_channels) x (H-2*kernel_size-2) x (W-2*kernel_size-2)
        x_skip = x.clone()
        x_skip = self.center_crop(x_skip) # B x (out_channels) x crop_size x crop_size
        x = self.max_pool(x) # B x (out_channels) x ((H-2*kernel_size-2)//2 + 1) x ((W-2*kernel_size - 2)//2 + 1)

        return x, x_skip
    
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.up_conv_layer_1 = nn.Conv2d(in_channels=self.in_channels, 
                                         out_channels=self.out_channels, 
                                         kernel_size=self.kernel_size)
        
        self.conv_layer_2 = nn.Conv2d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels,
                                        kernel_size=self.kernel_size)
        
        self.conv_layer_3 = nn.Conv2d(in_channels=self.out_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x, skip_input):

        x = self.up_conv_layer_1(x) # B x out_channels x H x W
        x = torch.concat([skip_input, x], dim = 1) # B x (2*out_channels) x H x W
        x = self.relu(self.conv_layer_2(x)) # B x out_channels x (H-kernel_size-1) x (W-kernel_size-1)
        x = self.relu(self.conv_layer_3(x)) # B x out_channels x (H-2*kernel_size-3) x (W-2*kernel_size-2)

        return x
    
class UNetMidBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.relu = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(in_channels=in_channels, 
                                      out_channels=2*self.out_channels, 
                                      kernel_size=self.kernel_size)
        
        self.conv_layer_2 = nn.Conv2d(in_channels=2*self.out_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=self.kernel_size)

    def forward(self, x):
        x = self.relu(self.conv_layer_1(x))
        x = self.relu(self.conv_layer_2(x))

        return x
    
class UNetEncoder(nn.Module):

    # Probably a more slick way of doing this with ModuleList

    def __init__(self, in_channels, out_channels, kernel_size, crop_sizes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.crop_sizes = crop_sizes

        self.encoder_block_1 = UNetEncoder(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=kernel_size,
                                      crop_size=self.crop_sizes[0])
        
        self.encoder_block_2 = UNetEncoder(in_channels=self.out_channels,
                                      out_channels=self.out_channels*2,
                                      kernel_size=self.kernel_size,
                                      crop_size=self.crop_sizes[1])
        
        self.encoder_block_3 = UNetEncoder(in_channels=self.out_channels*2,
                                      out_channels=self.out_channels*4,
                                      kernel_size=self.kernel_size,
                                      crop_size=self.crop_sizes[2])
        
        self.encoder_block_4 = UNetEncoder(in_channels=self.out_channels*4,
                                      out_channels=self.out_channels*8,
                                      kernel_size=self.kernel_size,
                                      crop_size=self.crop_sizes[3])

    def forward(self, x):
        x, x_skip_1 = self.encoder_block_1(x)
        x, x_skip_2 = self.encoder_block_2(x)
        x, x_skip_3 = self.encoder_block_3(x)
        x, x_skip_4 = self.encoder_block_4(x)

        return x, [x_skip_1, x_skip_2, x_skip_3, x_skip_4]
    
class UNetDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, crop_sizes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.crop_sizes = crop_sizes

        self.decoder_block_1 = UNetDecoderBlock(in_channels=self.in_channels,
                                                out_channels=self.out_channels,
                                                kernel_size=self.kernel_size)
        
        self.decoder_block_2 = UNetDecoderBlock(in_channels=self.out_channels,
                                                out_channels=self.out_channels//2,
                                                kernel_size=self.kernel_size)
        
        self.decoder_block_3 = UNetDecoderBlock(in_channels=self.out_channels,
                                        out_channels=self.out_channels//4,
                                        kernel_size=self.kernel_size)
        
        self.decoder_block_4 = UNetDecoderBlock(in_channels=self.out_channels,
                                out_channels=self.out_channels//8,
                                kernel_size=self.kernel_size)

    def forward(self, x, skip_connections):

        x = self.decoder_block_1(x, skip_connections[0])
        x = self.decoder_block_2(x, skip_connections[1])
        x = self.decoder_block_3(x, skip_connections[2])
        x = self.decoder_block_4(x, skip_connections[3])
        
        return x
