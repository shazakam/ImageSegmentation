import torch
import pytest
from models.model_blocks import (
    UNetDecoderBlock,
    UNetEncoderBlock,
    UNetMidBlock,
    UNetDecoder,
    UNetEncoder
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

def test_mid_block_forward_x_out():
    in_channels = 512
    out_channels = 512
    kernel_size = 3
    encoder = UNetMidBlock(in_channels,out_channels,kernel_size)
    test_encoder_input = torch.randn((16,in_channels,32,32))

    x = encoder(test_encoder_input) 
    assert(x.shape == (16, out_channels, 28, 28)) 

def test_decoder_block_forward_x_out():
    in_channels = 1024
    out_channels = 512
    kernel_size = 3
    decoder = UNetDecoderBlock(in_channels, out_channels, kernel_size)
    x_skip = torch.randn((16,512,56,56))
    x = torch.randn((16, 1024,28,28))

    x = decoder(x, x_skip)
    assert(x.shape == (16, out_channels, 52, 52)) 

def test_encoder_forward_x_out():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    crop_sizes = [392, 200, 104, 56]
    x = torch.randn((16, in_channels, 572, 572))
    UNEncoder = UNetEncoder(in_channels, out_channels, kernel_size, crop_sizes)
    x, _ = UNEncoder(x)

    assert(x.shape == (16, 512, 32, 32)) 

def test_encoder_forward_x_skip_out():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    crop_sizes = [392, 200, 104, 56]
    x = torch.randn((16, in_channels, 572, 572))
    UNEncoder = UNetEncoder(in_channels, out_channels, kernel_size, crop_sizes)
    _, x_skip = UNEncoder(x)

    assert(x_skip[0].shape == (16, 64, 392, 392)) 
    assert(x_skip[1].shape == (16, 128, 200, 200))
    assert(x_skip[2].shape == (16, 256, 104, 104))
    assert(x_skip[3].shape == (16, 512, 56, 56))

def test_decoder_forward_x_out(): 
    in_channels = 1024
    out_channels = 512
    kernel_size = 3

    x_skip = [torch.randn((16, 64, 392, 392)), 
              torch.randn((16, 128, 200, 200)), 
              torch.randn((16, 256, 104, 104)), 
              torch.randn((16, 512, 56, 56))]
    
    UNDecoder = UNetDecoder(in_channels, out_channels, kernel_size)
    x = torch.randn((16, in_channels, 28, 28))

    x = UNDecoder(x, x_skip)
    assert(x.shape == (16, 64, 388, 388)) 
    

def test_unet_forward():

    unet = UNet(in_channels=3, 
                out_channels=64,
                kernel_size=3, 
                crop_sizes=[392, 200, 104, 56], 
                final_filters=2)

    x = torch.randn((16, 3, 572, 572))
    x = unet(x)
    assert(x.shape == (16, 2, 388, 388))



