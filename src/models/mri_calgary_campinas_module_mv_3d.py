import json
import os
from os.path import isfile, join
from typing import Any, Dict, Tuple, Callable, List, Optional

import torch
import torchmetrics
import wandb
from lightning import LightningModule
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics import MaxMetric, MeanMetric

from src.utils.io_utils import save_tensor_to_nifti


class MRI_Calgary_Campinas_LitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        train_acc: Dict = {'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure},
        val_acc: Dict = {'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure},
        test_acc: Dict = {'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure},
        criterions: Dict = {'mse': torch.nn.MSELoss},
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss functions
        self.criterions = criterions

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = nn.ModuleDict(train_acc)
        self.val_acc = nn.ModuleDict(val_acc)
        self.test_acc = nn.ModuleDict(test_acc)
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = {}#[MaxMetric() for x in self.test_acc]
        for key ,acc in self.val_acc.items():
            self.val_acc_best[key] = MaxMetric()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        for key ,acc in self.val_acc.items():
            acc.reset()
        for key ,acc in self.val_acc_best.items():
            acc.reset()


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """


        output_image, output_kspace, target_img, output_image_mv,x_volume = self.forward(batch)
        # target_img = torch.abs(batch["target"]).squeeze(1)

        loss = {}

        for key, criterion in self.criterions.items():
            if key == 'l1':
                loss[key] = criterion(output_image.unsqueeze(1), target_img.unsqueeze(1))
            else:
                loss[key] = criterion(x_volume.unsqueeze(1), target_img.unsqueeze(1))
            # if key == 'lncc':
            #     loss[key] = (0.99 ** self.current_epoch) * criterion(x_volume.unsqueeze(1), target_img.unsqueeze(1))
            # else:
            #     loss[key] = criterion(output_image.unsqueeze(1), target_img.unsqueeze(1))
            # loss[key] = criterion(x_volume.unsqueeze(1), target_img.unsqueeze(1), output_image.unsqueeze(1), self.current_epoch)

        return loss, output_image, target_img,output_image_mv,x_volume

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.tele
        :return: A tensor of losses between model predictions and targets.
        """
        losses , preds, targets, output_image_sv,_ = self.model_step(batch)

        for key, acc in self.train_acc.items():
            acc(preds.unsqueeze(1), targets.unsqueeze(1))
            self.log(f"train_acc/{key}", acc, on_step=False, on_epoch=True, prog_bar=False)

        for key, loss in losses.items():
            # self.train_loss(loss)
            self.log(f"train_loss/{key}", loss, on_step=True, on_epoch=True, prog_bar=False)


        # return loss or backpropagation will fail
        return sum(losses.values())

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, targets, output_image_svs,x_volume  = self.model_step(batch)
        initial = preds - output_image_svs
        if (self.current_epoch % 5 == 0 and batch_idx == 0):
            # columns = [ 'prediction','ground truth']
            n = 0
            # data = [[wandb.Image(x_i), wandb.Image(y_i)] for x_i, y_i in list(zip(preds[:n], targets[:n]))]
            # self.logger.log_table(key='Comparison', columns=columns, data=data)
            # for n in range(2):
            n = preds.shape[0]//2
            m = preds.shape[1]//2
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figsize as needed
            # print(preds.shape)
            pred =(preds[n,m]/preds[n,m].max()).cpu().detach()
            # Plot prediction
            im0 = axs[0,0].imshow(pred,cmap='gray')  # Assuming preds[i] is a 2D array or an image file
            axs[0,0].title.set_text(f'Prediction in epoch: {self.current_epoch}')
            fig.colorbar(im0, ax=axs[0,0])
            axs[0,0].axis('off')  # Hide axis

            target = (targets[n,m]/targets[n,m].max()).cpu().detach()
            # Plot ground truth
            im1 = axs[0,1].imshow(target,cmap='gray')  # Assuming targets[i] is a 2D array or an image file
            axs[0,1].title.set_text('Ground Truth')
            fig.colorbar(im1, ax=axs[0,1])
            axs[0,1].axis('off')

            im2 = axs[0,2].imshow(torch.abs(pred-target),cmap='gray')  # Assuming targets[i] is a 2D array or an image file
            axs[0,2].title.set_text('Diff')
            axs[0,2].axis('off')
            fig.colorbar(im2, ax=axs[0,2])

            output_image_sv = (output_image_svs[n,m] / output_image_svs[n,m].max()).cpu().detach()
            im3 = axs[1,0].imshow(output_image_sv,cmap='gray')  # Assuming output_image_sv[i] is a 2D array or an image file
            axs[1,0].title.set_text('Res')
            axs[1,0].axis('off')
            fig.colorbar(im3, ax=axs[1,0])

            img_pre = (x_volume[n,m] / x_volume[n,m].max()).cpu().detach()
            im3 = axs[1,1].imshow(img_pre,cmap='gray')  # Assuming output_image_sv[i] is a 2D array or an image file
            axs[1,1].title.set_text('Previous')
            axs[1,1].axis('off')
            fig.colorbar(im3, ax=axs[1,1])

            initial = (initial[n,m] / initial[n,m].max()).cpu().detach()
            im3 = axs[1,2].imshow(initial,cmap='gray')  # Assuming output_image_sv[i] is a 2D array or an image file
            axs[1,2].title.set_text('Initial rec')
            axs[1,2].axis('off')
            fig.colorbar(im3, ax=axs[1,2])

            self.logger.log_image(key="samples", images=[fig])
            plt.close()



        # update and log metrics
        for key, acc in self.val_acc.items():
            B, Z, H, W = preds.shape
            acc(preds.view(B*Z, 1, H, W), targets.view(B*Z, 1, H, W))
            self.log(f"val_acc/{key}", acc, on_step=False, on_epoch=True, prog_bar=False)

        for key, loss in losses.items():
            # self.val_loss(loss)
            self.log(f"val_loss/{key}", loss, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        for key, acc in self.val_acc.items():
            acc = acc.compute()  # get current val acc
            self.val_acc_best[key](acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
            self.log(f"val_acc_best/{key}", self.val_acc_best[key].compute(), sync_dist=True, prog_bar=False)
            # self.val_acc_best.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        print(batch["data"].shape)
        losses, preds, targets, output_image_mv, _ = self.model_step(batch)
        # preds = preds - output_image_mv
        print(preds.shape)
        save_tensor_to_nifti(preds.squeeze(), join(self.logger.save_dir, f"{batch['metadata']['File name'][0]}_preds.nii"))
        save_tensor_to_nifti(targets.squeeze(), join(self.logger.save_dir, f"{batch['metadata']['File name'][0]}_targets.nii"))
        accuracies = {}
        for key, acc in self.test_acc.items():
            # acc_ = []
            # for i in range(preds.shape[0]):
            #     acc(preds[i, None, None, ...], targets[i, None, None, ...])
            #     acc_.append(acc.compute())
            
            # if key not in accuracies:
            #     accuracies[key] = []

            acc(preds.unsqueeze(0), targets.unsqueeze(0))
            # accuracies[key] = [(x.cpu().numpy()).tolist() for x in acc_]
            # self.log(f"test_acc/{key}_mean", torch.mean(torch.Tensor(acc_)), on_step=True, on_epoch=True, prog_bar=False)
            # self.log(f"test_acc/{key}_std", torch.std(torch.Tensor(acc_)), on_step=True, on_epoch=True, prog_bar=False)
            self.log(f"test_acc/{key}_whole", acc.compute(), on_step=True, on_epoch=True, prog_bar=False)
            acc.reset()

        # nmse = src.utils.fastMRI.evaluate.nmse(targets.unsqueeze(1),preds.unsqueeze(1))
        # self.log(f"test_acc/{key}_nmse", nmse, on_step=True, on_epoch=True, prog_bar=False)

        # ssim = calgary_capinas_ssim(targets.unsqueeze(1),preds.unsqueeze(1))
        # self.log(f"test_acc/{key}_nmse", ssim, on_step=True, on_epoch=True, prog_bar=False)

        # psnr = src.utils.fastMRI.evaluate.psnr(targets.unsqueeze(1),preds.unsqueeze(1))
        # self.log(f"test_acc/{key}_nmse", psnr, on_step=True, on_epoch=True, prog_bar=False)

        # print("micky mouse")
            
        # update and log metrics
        # self.log('loss', loss)
        for key, loss in losses.items():
            # self.test_loss(loss)
            self.log(f"test_loss/{key}", loss, on_step=True, on_epoch=True, prog_bar=False)


        # Define JSON file path
        json_file_path = join(self.logger.save_dir, 'accuracies.json')

        # Load existing data if file exists
        if isfile(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Update the JSON data with the new accuracies
        # The key is batch['metadata']['File name'][0]
        batch_key = batch['metadata']['File name'][0]
        data[batch_key] = accuracies

        # Write the updated dictionary back to the file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        torch.save(self.net.state_dict(), join(self.logger.save_dir,'model_weights.pth'))

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_acc/SSIM",
                    "interval": "epoch",
                    "frequency": 1,
                    "name" : "my scheduler",
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MRI_Calgary_Campinas_LitModule(None, None, None, None)
