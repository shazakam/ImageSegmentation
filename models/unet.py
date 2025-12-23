import torch
import torch.nn as nn
from models.model_blocks import UNetEncoder, UNetDecoder, UNetMidBlock
import lightning as L
from data_utils.metrics import binary_iou
class UNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, 
                 final_filters,
                 encoder_dropout):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.final_filters = final_filters
        self.encoder_dropout = encoder_dropout

        self.unet_encoder = UNetEncoder(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=self.kernel_size,
                                        dropout=self.encoder_dropout)
        
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
    
class ImageSegmentationModel(L.LightningModule):
    def __init__(self, model, loss_function, lr, scheduler_step, scheduler_gamma):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = lr
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step = scheduler_step

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y, y_hat)
        self.log("train_loss", loss, on_epoch=True, on_step=True,prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y, y_hat)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y, y_hat)
        test_metric = binary_iou(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_biou", test_metric)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=self.scheduler_gamma,
                        patience=self.scheduler_step,
                    )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }