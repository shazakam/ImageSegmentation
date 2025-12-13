from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import wandb
from models.unet import UNet
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import yaml
from data_utils.pre_process import load_datasets, load_images_and_labels
import albumentations as A
from torch.utils.data import DataLoader
def load_sweep_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Sweep training entry point
def train_sweep(project, train_dataset, val_dataset):

    with wandb.init(project=project) as run:
        config = wandb.config
        # GET ALL NECESSARY HPARAMS OUT OF CONFIG AND PASS THEM

        model = UNet(in_channels=3, 
                out_channels=64,
                kernel_size=3, 
                crop_sizes=[392, 200, 104, 56], 
                final_filters=1)

        logger = WandbLogger(project="UNet-Image-Segmentation")

        callbacks = [EarlyStopping(monitor="val_loss", mode="min"), 
                    LearningRateMonitor(logging_interval='step'),
                    ModelCheckpoint(
                    monitor="val_loss",      # metric to monitor
                    mode="min",              # "min" for loss, "max" for accuracy/IoU
                    save_top_k=1,            # save ONLY the best model
                    save_last=False,         # don't save last epoch
                    filename="best-{epoch}-{val_loss:.4f}",
                )]
        
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=config['batch_size'],
                                      shuffle=True)
        
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle=False)
        
        trainer = Trainer(
            logger=logger,
            max_epochs=config.epochs, # Epochs in config
            callbacks=callbacks
        )

        trainer.fit(model,train_dataloader, val_dataloader)

wandb_project = "unet-image-segmentation-sweeps"
sweep_config_path = "sweep/sweep_config.yaml"
input_folder_path = "oxford-iiit-pet/images"
label_folder_path = "oxford-iiit-pet/annotations/trimaps"

wandb_config = load_sweep_config(sweep_config_path)

input_images, label_images = load_images_and_labels(input_folder_path=input_folder_path,
                                                    label_folder_path=label_folder_path)

train_transforms = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),                  
    A.RandomCrop(height=256, width=256), 
    A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        normalization="standard", # Default
                        p=1.0),
    A.ToTensorV2()                            
])

val_test_transforms = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),                  
    A.RandomCrop(height=256,width=256),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                normalization="standard", # Default
                p=1.0),
    A.ToTensorV2()                          
])

train_dataset, val_dataset, test_dataset = load_datasets(input_images, 
                                                            label_images, 
                                                            train_transforms,
                                                            val_test_transforms,
                                                            save_path="saved_data",
                                                            shuffle=True)
# Instantiate Sweep
sweep_id = wandb.sweep(sweep=wandb_config, project=wandb_project)

# Start sweep with agent
wandb.agent(sweep_id=sweep_id, function=lambda x : train_sweep(wandb_project, 
                                                               train_dataset, 
                                                               val_dataset))
