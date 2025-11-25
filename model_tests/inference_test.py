import torch
import pytest
from models.model_blocks import (
    UNetDecoderBlock,
    UNetEncoderBlock,
    UNetMidBlock,
    UNetDecoder,
    UNetEncoder,
)
from models.unet import UNet

def test_encoder_block_forward_x_out():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    crop_sizes = 392
    encoder = UNetEncoderBlock(in_channels,out_channels, kernel_size, crop_sizes)
    test_encoder_input = torch.randn((16,in_channels,572,572))

    x, _ = encoder(test_encoder_input) 
    assert(x.shape == (16, out_channels, 284,284))

def test_encoder_block_forward_x_skip_out():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    crop_size = 392
    encoder = UNetEncoderBlock(in_channels, out_channels, kernel_size, crop_size)
    test_encoder_input = torch.randn((16, in_channels, 572, 572))

    _, x_skip = encoder(test_encoder_input) 
    assert(x_skip.shape == (16, out_channels, 392,392)) 

# TODO: UNet Mid Test
def test_mid_block_forward_x_skip_out():
    in_channels = 512
    out_channels = 512
    kernel_size = 3
    encoder = UNetMidBlock(in_channels,out_channels,kernel_size)
    test_encoder_input = torch.randn((16,in_channels,32,32))

    x = encoder(test_encoder_input) 
    assert(x.shape == (16, out_channels, 56, 56)) 

# TODO: UNet Decoder Test

# TODO: UNet Inference Test
