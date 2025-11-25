import torch
import torch.nn as nn
from model_blocks import UNetEncoder, UNetDecoder, UNetMidBlock


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, crop_sizes, final_filters):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.crop_sizes = crop_sizes
        self.final_filters = final_filters

        self.unet_encoder = UNetEncoder(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=self.kernel_size,
                                        crop_sizes=self.crop_sizes)
        
        self.unet_midblock = UNetMidBlock(in_channels=self.out_channels*8, 
                                          out_channels=self.out_channels*16, 
                                          kernel_size=self.kernel_size)
        
        self.unet_decoder = UNetDecoder(in_channels=self.out_channels*8, out_channels=self.out_channels*4)

        self.final_conv = nn.Conv2d(in_channels=self.out_channels,
                               out_channels=self.final_filters,
                               kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.unet_encoder(x) # Input: (B, C, H, W)
        x = self.unet_midblock(x)
        x = self.unet_decoder(x, skip_connections)
        x =  self.final_conv(x)

        return x