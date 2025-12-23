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
    encoder = UNetEncoderBlock(in_channels,out_channels, kernel_size)
    test_encoder_input = torch.randn((16,in_channels,512,512))

    x, _ = encoder(test_encoder_input) 
    assert(x.shape == (16, out_channels, 256,256))

def test_encoder_block_forward_x_skip_out():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    encoder = UNetEncoderBlock(in_channels, out_channels, kernel_size)
    test_encoder_input = torch.randn((16, in_channels, 512, 512))

    _, x_skip = encoder(test_encoder_input) 
    assert(x_skip.shape == (16, out_channels, 512,512)) 

def test_mid_block_forward_x_out():
    in_channels = 512
    out_channels = 512
    kernel_size = 3
    encoder = UNetMidBlock(in_channels,out_channels,kernel_size)
    test_encoder_input = torch.randn((16,in_channels,32,32))

    x = encoder(test_encoder_input) 
    assert(x.shape == (16, out_channels, 32, 32)) 

def test_decoder_block_forward_x_out():
    in_channels = 1024
    out_channels = 512
    kernel_size = 3
    decoder = UNetDecoderBlock(in_channels, out_channels, kernel_size)
    x_skip = torch.randn((16,512,64,64))
    x = torch.randn((16, 1024,32,32))

    x = decoder(x, x_skip)
    assert(x.shape == (16, out_channels, 64, 64)) 

def test_encoder_forward_x_out():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    x = torch.randn((16, in_channels, 512, 512))
    UNEncoder = UNetEncoder(in_channels, out_channels, kernel_size)
    x, _ = UNEncoder(x)

    assert(x.shape == (16, 512, 32, 32)) 

def test_encoder_forward_x_skip_out():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    x = torch.randn((16, in_channels, 512, 512))
    UNEncoder = UNetEncoder(in_channels, out_channels, kernel_size)
    _, x_skip = UNEncoder(x)

    assert(x_skip[0].shape == (16, 64, 512, 512)) 
    assert(x_skip[1].shape == (16, 128, 256, 256))
    assert(x_skip[2].shape == (16, 256, 128, 128))
    assert(x_skip[3].shape == (16, 512, 64, 64))

def test_decoder_forward_x_out(): 
    in_channels = 1024
    out_channels = 512
    kernel_size = 3

    x_skip = [torch.randn((16, 64, 512, 512)), 
              torch.randn((16, 128, 256, 256)), 
              torch.randn((16, 256, 128, 128)), 
              torch.randn((16, 512, 64, 64))]
    
    UNDecoder = UNetDecoder(in_channels, out_channels, kernel_size)
    x = torch.randn((16, in_channels, 32, 32))

    x = UNDecoder(x, x_skip)
    assert(x.shape == (16, 64, 512, 512)) 
    

def test_unet_forward():

    unet = UNet(in_channels=3, 
                out_channels=64,
                kernel_size=3, 
                final_filters=1,
                encoder_dropout=0.1)

    x = torch.randn((16, 3, 512, 512))
    x = unet(x)
    assert(x.shape == (16, 1, 512, 512))



