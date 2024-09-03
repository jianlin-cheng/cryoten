import glob
import os

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.cryoem_map_datasets import (
    CryoemDensityMapBlockAugmentedDataset, # Train
    CryoemDensityMapBlockDataset, # Validation
    CryoemDensityMapBlockPredictDataset, # Prediction
)

from monai.transforms import (
    Compose,
    RandRotate90,
    RandAxisFlip,
    RandSpatialCrop,
)


class CryoemDensityMapDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir="",
        predict_dataset_dir="",
        batch_size=8,
        num_workers=1,
        train_max_samples=10000,
        val_max_samples=1000,
        pin_memory=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset_dir = dataset_dir
        self.predict_dataset_dir = predict_dataset_dir
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_max_samples = train_max_samples
        self.val_max_samples = val_max_samples
        self.pin_memory = pin_memory

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.train_input_maps_data_dir = os.path.join(self.dataset_dir, "train", "input")
            self.train_target_maps_data_dir = os.path.join(self.dataset_dir, "train", "target")
            self.val_input_maps_data_dir = os.path.join(self.dataset_dir, "val", "input")
            self.val_target_maps_data_dir = os.path.join(self.dataset_dir, "val", "target")
            
            list_of_train_input_images = [
                os.path.basename(x)
                for x in glob.glob(self.train_input_maps_data_dir + "/*.mrc")
            ]
            list_of_train_target_images = [
                os.path.basename(x)
                for x in glob.glob(self.train_target_maps_data_dir + "/*.mrc")
            ]
            list_of_val_input_images = [
                os.path.basename(x)
                for x in glob.glob(self.val_input_maps_data_dir + "/*.mrc")
            ]
            list_of_val_target_images = [
                os.path.basename(x)
                for x in glob.glob(self.val_target_maps_data_dir + "/*.mrc")
            ]

            if len(list_of_train_input_images) != len(list_of_train_target_images):
                print("ERROR: No of training input blocks and target blocks not same!")
                return

            if len(list_of_val_input_images) != len(list_of_val_target_images):
                print("ERROR: No of validation input blocks and target blocks not same!")
                return

            # list_of_train_input_images.sort()
            if self.train_max_samples != 0 and self.train_max_samples < len(
                list_of_train_input_images
            ):
                list_of_train_input_images = list_of_train_input_images[
                    : self.train_max_samples
                ]

            list_of_val_input_images.sort()
            if self.val_max_samples != 0 and self.val_max_samples < len(
                list_of_val_input_images
            ):
                list_of_val_input_images = list_of_val_input_images[: self.val_max_samples]
            
            i_transform = Compose(
                [
                    RandSpatialCrop(
                        [48, 48, 48],
                        max_roi_size=[48, 48, 48],
                        random_center=True,
                        random_size=False,
                    ),
                    RandRotate90(prob=0.4),
                    RandAxisFlip(prob=0.4),
                ]
            )

            t_transform = Compose(
                [
                    RandSpatialCrop(
                        [48, 48, 48],
                        max_roi_size=[48, 48, 48],
                        random_center=True,
                        random_size=False,
                    ),
                    RandRotate90(prob=0.4),
                    RandAxisFlip(prob=0.4),
                ]
            )

            self.train_set = CryoemDensityMapBlockAugmentedDataset(
                list_of_train_input_images,
                self.train_input_maps_data_dir,
                self.train_target_maps_data_dir,
                i_transform,
                t_transform,
            )
            self.val_set = CryoemDensityMapBlockDataset(
                list_of_val_input_images,
                self.val_input_maps_data_dir,
                self.val_target_maps_data_dir,
            )
        elif stage == "predict":
            self.predict_set = CryoemDensityMapBlockPredictDataset(self.predict_dataset_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # persistent_workers = True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # persistent_workers = True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # persistent_workers = True
        )
