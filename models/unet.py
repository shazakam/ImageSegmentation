import torch
import torch.nn as nn
from models.model_blocks import UNetEncoder, UNetDecoder, UNetMidBlock
import pytorch_lightning as pl

class UNet(pl.Module):
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
        
        self.unet_decoder = UNetDecoder(in_channels=self.out_channels*16, 
                                        out_channels=self.out_channels*8,
                                        kernel_size=self.kernel_size)

        self.final_conv = nn.Conv2d(in_channels=self.out_channels,
                               out_channels=self.final_filters,
                               kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.unet_encoder(x) # Input: (B, C, H, W)
        x = self.unet_midblock(x)
        x = self.unet_decoder(x, skip_connections)
        
        x =  self.final_conv(x)

        return x
    
class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, model, loss_function, optimiser, lr, scheduler_step, scheduler_gamma):
        super().__init__()
        self.model = model
        self.loss_function = loss_function

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_function(y, y_hat)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_function(y, y_hat)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_function(y, y_hat)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"    # required for ReduceLROnPlateau
        }