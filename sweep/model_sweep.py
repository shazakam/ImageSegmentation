from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import wandb
from models.unet import UNet, ImageSegmentationModel
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import yaml
from data_utils.pre_process import load_datasets, load_images_and_labels
import albumentations as A
from torch.utils.data import DataLoader
import torch
#kljlkjlk
def load_sweep_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Sweep training entry point
def train_sweep(project, train_dataset, val_dataset, test_dataset):

    with wandb.init(project=project) as run:
        config = wandb.config
    
    # GET ALL NECESSARY HPARAMS OUT OF CONFIG AND PASS THEM

    unet = UNet(in_channels=3, 
            out_channels=64,
            kernel_size=3, 
            final_filters=1,
            encoder_dropout=config.encoder_dropout)
    
    loss_criterion = torch.nn.BCEWithLogitsLoss()
    
    segmentation_model = ImageSegmentationModel(model = unet,
                                                loss_function = loss_criterion,
                                                lr = config.learning_rate,
                                                scheduler_step = config.scheduler_step,
                                                scheduler_gamma = config.scheduler_gamma)
    
    logger = WandbLogger(log_model=True)

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
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=1,
                                    persistent_workers=True)
    
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                persistent_workers=True)
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 persistent_workers=True)
    
    trainer = Trainer(
        logger=logger,
        max_epochs=config.max_epochs, # Epochs in config
        callbacks=callbacks,
        accelerator="mps",
        log_every_n_steps=config.batch_size
    )

    trainer.fit(segmentation_model, train_dataloader, val_dataloader)

    test_results = trainer.test(ImageSegmentationModel, test_dataloader)
    test_iou = test_results[0]["test_iou"]

    run.log({"test_iou", test_iou})


if __name__ == "__main__":
    wandb_project = "unet-image-segmentation-sweeps"
    sweep_config_path = "sweep/sweep_config.yaml"
    input_folder_path = "oxford-iiit-pet/images"
    label_folder_path = "oxford-iiit-pet/annotations/trimaps"

    print("Loading Sweep Configurations")
    wandb_config = load_sweep_config(sweep_config_path)

    print("Loading Images")
    input_images, label_images = load_images_and_labels(input_folder_path=input_folder_path,
                                                        label_folder_path=label_folder_path)
    input_images = input_images[:]
    label_images = label_images[:]

    print("Initialising Transforms")
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

    print("Creating Datasets")
    train_dataset, val_dataset, test_dataset = load_datasets(input_images, 
                                                                label_images, 
                                                                train_transforms,
                                                                val_test_transforms,
                                                                save_path="saved_data",
                                                                shuffle=True)
    
    print("Instantiating sweep")
    # Instantiate Sweep
    sweep_id = wandb.sweep(sweep=wandb_config, project=wandb_project)

    print("Starting Sweep")
    # Start sweep with agent
    wandb.agent(sweep_id=sweep_id, 
                function=lambda : train_sweep(wandb_project, 
                                            train_dataset, 
                                            val_dataset,
                                            test_dataset),
                count = 3) 
