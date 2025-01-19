import glob
from typing import Any, Dict, Optional, Tuple, List, Callable

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import pandas as pd

from src.data.components.calgary_campinas_dataset_mv_roberto_3d import SliceDataset


class C2DataModule(LightningDataModule):
    """`
    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir : str = None,
        metadata_train_dir : str = None,
        metadata_val_dir: str = None,
        metadata_test_dir: str = None,
        target_dir: str = None,
        baseline_dir: str = None,
        transforms_input: Optional[Callable] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        crop_slice_idx = 0,
        special_transforms: Optional[Callable] = None,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms_input = transforms.Compose(
            transforms_input
        )
        # data transformations
        self.special_transforms = transforms.Compose(
            special_transforms
        )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # self.metadata = pd.read_csv(self.hparams.metadata_dir)
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

        # if self.hparams.smap_style == 'circle_ring':
        #     self.masks = [f'{self.hparams.mask_dir}/uniform_mask_R=6.npy',
        #                   f'{self.hparams.mask_dir}/uniform_mask_R=8.npy']
        # else:
        #     self.masks = [f'{self.hparams.mask_dir}/uniform_mask_R=2.npy',
        #                   f'{self.hparams.mask_dir}/218_170/uniform_mask_R=4.npy']
        # self.data_train = SliceDataset(self.hparams.data_dir, self.slice_ids, 'train', self.smaps, self.masks, 'nlinv',
        #                                self.hparams.coils,
        #                                data_transforms=self.transforms, target_transforms=self.transforms)
        # self.data_val = SliceDataset(self.hparams.data_dir, self.slice_ids_val, 'val', self.smaps, self.masks, 'nlinv',
        #                              self.hparams.coils,
        #                              data_transforms=self.transforms, target_transforms=self.transforms)
        # self.data_test = SliceDataset(self.hparams.data_dir, self.slice_ids_val, 'val', self.smaps, self.masks, 'nlinv',
        #                               self.hparams.coils,
        #                               data_transforms=self.transforms, target_transforms=self.transforms)



    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:

        self.data_train = SliceDataset(self.hparams.data_dir,
                                       self.hparams.target_dir,
                                       self.hparams.baseline_dir,
                                       self.hparams.metadata_train_dir,
                                       input_transforms=self.transforms_input,
                                       crop_slice_idx = self.hparams.crop_slice_idx,
                                        special_transforms = self.special_transforms)
        self.data_val = SliceDataset(self.hparams.data_dir,
                                     self.hparams.target_dir,
                                     self.hparams.baseline_dir,
                                       self.hparams.metadata_val_dir,
                                     input_transforms=self.transforms_input,
                                       crop_slice_idx = self.hparams.crop_slice_idx)
        self.data_test = SliceDataset(self.hparams.data_dir,
                                      self.hparams.target_dir,
                                      self.hparams.baseline_dir,
                                       self.hparams.metadata_test_dir,
                                      input_transforms=self.transforms_input,
                                       crop_slice_idx = self.hparams.crop_slice_idx)


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,#256-self.hparams.crop_slice_idx*2,#
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,#256-self.hparams.crop_slice_idx*2,#self.batch_size_per_device,#256-self.hparams.crop_slice_idx*2,# # TODO: make 256 felixible
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

@hydra.main(version_base="1.3", config_path="../../configs/data/", config_name="mri_calgary_campinas.yaml")
def main(cfg: DictConfig):
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
if __name__ == "__main__":
    main()
