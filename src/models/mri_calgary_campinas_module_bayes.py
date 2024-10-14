import json
import os
from os.path import isfile, join
from typing import Any, Dict, Tuple, Callable, List, Optional
import copy
import torch
import torchmetrics
import wandb
from lightning import LightningModule
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics import MaxMetric, MeanMetric

from src.utils.io_utils import save_tensor_to_nifti
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.utils.util import predictive_entropy, mutual_information
from bayesian_torch.layers.variational_layers import Conv2dReparameterization
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
        net_path: None,
        num_of_infer,
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
        self.num_of_infer = num_of_infer
        moped_enable_ = False
        if net_path is not None:
            self.net.load_state_dict(torch.load(net_path))
            print("a pretrained model is loaded")
            moped_enable_ = True

        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",  # Flipout or Reparameterization
            "moped_enable": moped_enable_,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.005,
        }

        dnn_to_bnn(self.net, const_bnn_prior_parameters)
        
        self.bayes_conv_std = Conv2dReparameterization(
            in_channels=1, 
            out_channels=1,
            kernel_size=1,
            prior_mean=0.0, 
            prior_variance=1.0, 
            posterior_mu_init=0.0, 
            posterior_rho_init=-3.0
        )



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


        output_image, output_kspace, target_img = self.forward(batch)
        # print(output_image.shape)
        # Check if the output_image is 3-dimensional
        num_dims = output_image.dim()
        if num_dims == 3:
            # Expand the dimensions along the first axis
            output_image = output_image.unsqueeze(1)
        # Predict the mean
        # mean = self.bayes_conv_mean(output_image, return_kl=False)
        # Predict the standard deviation
        bayes_conv_out = self.bayes_conv_std(output_image, return_kl=False)
        # mean, std = torch.split(bayes_conv_out, 1, dim=1)
        std = torch.exp(bayes_conv_out)  # Ensure the std is positive
        if num_dims == 3:
            # Expand the dimensions along the first axis
            output_image = output_image.squeeze(1)
            std = std.squeeze(1)
        # output_image = mean
        # output_image = self.conv_final(output_image)
        # print(output_image.shape)
        # target_img = torch.abs(batch["target"]).squeeze(1)

        loss = {}
        if self.criterions != None:
            for key, criterion in self.criterions.items():
                loss[key] = criterion(output_image, target_img)
        loss['nll'] = self.nll_loss(target_img, output_image, std)
        loss['bayes'] = (get_kl_loss(self.net)/output_image.shape[0])
        return loss, output_image, target_img, std
    def nll_loss(self,y_true, y_pred_mean, y_pred_std):
        variance = y_pred_std ** 2
        loss = torch.mean(torch.log(variance) + (y_true - y_pred_mean) ** 2 / (2 * variance))
        return loss
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses , preds, targets, std = self.model_step(batch)

        for key, acc in self.train_acc.items():
            acc(preds.unsqueeze(1), targets.unsqueeze(1))
            self.log(f"train_acc/{key}", acc, on_step=False, on_epoch=True, prog_bar=False)

        for key, loss in losses.items():
            # self.train_loss(loss)
            self.log(f"train_loss/{key}", loss, on_step=False, on_epoch=True, prog_bar=False)


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
        # losses, preds, targets = self.model_step(batch)

        # Initialize empty lists to store results
        all_preds = []
        all_aleatoric_uncertainties = []

        # Run the function `num_of_infer` times to gather epistemic uncertainty
        for _ in range(self.num_of_infer):
            losses, preds, targets, std = self.model_step(copy.deepcopy(batch))
            
            # Append results to lists
            all_preds.append(preds)
            all_aleatoric_uncertainties.append(std)  # Aleatoric uncertainty (std from the model)

        # Stack the lists along a new dimension (assuming preds and targets are tensors with the same shape)
        all_preds = torch.stack(all_preds)
        all_aleatoric_uncertainties = torch.stack(all_aleatoric_uncertainties)
        # predictive_uncertainty = predictive_entropy(all_preds.cpu().numpy())
        # model_uncertainty = mutual_information(all_preds.cpu().numpy())

        # Epistemic uncertainty (variance of predictions)
        variance_preds = all_preds.var(dim=0)
        # Mean prediction
        preds = all_preds.mean(dim=0)
        # Mean aleatoric uncertainty
        aleatoric_uncertainty = all_aleatoric_uncertainties.mean(dim=0)


        if (self.current_epoch % 5 == 0 and batch_idx == 10):
            n = preds.shape[0] // 2  # Example to pick an image to visualize

            fig, axs = plt.subplots(1, 5, figsize=(18, 5))  # Adjusted figsize as needed
            pred = (preds[n] / preds[n].max()).cpu().detach()

            # Plot prediction
            im0 = axs[0].imshow(pred, cmap='gray')
            axs[0].title.set_text(f'Prediction in epoch: {self.current_epoch}')
            fig.colorbar(im0, ax=axs[0])
            axs[0].axis('off')  # Hide axis

            target = (targets[n] / targets[n].max()).cpu().detach()
            # Plot ground truth
            im1 = axs[1].imshow(target, cmap='gray')
            axs[1].title.set_text('Ground Truth')
            fig.colorbar(im1, ax=axs[1])
            axs[1].axis('off')

            im2 = axs[2].imshow(torch.abs(pred - target), cmap='gray')
            axs[2].title.set_text('Diff')
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2])

            im3 = axs[3].imshow((variance_preds[n] / variance_preds[n].max()).cpu().detach(), cmap='gray')
            axs[3].title.set_text('Epistemic Uncertainty')
            axs[3].axis('off')
            fig.colorbar(im3, ax=axs[3])

            im4 = axs[4].imshow((aleatoric_uncertainty[n].squeeze() / aleatoric_uncertainty[n].max()).cpu().detach(), cmap='gray')
            axs[4].title.set_text('Aleatoric Uncertainty')
            axs[4].axis('off')
            fig.colorbar(im4, ax=axs[4])

            self.logger.log_image(key="samples", images=[fig])
            plt.close()



        # update and log metrics
        for key, acc in self.val_acc.items():
            acc(preds.unsqueeze(1), targets.unsqueeze(1))
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
        :param batch_idx: The index of the current batch. """
        # Initialize empty lists to store results
        all_preds = []
        all_aleatoric_uncertainties = []

        # Run the function `num_of_infer` times to gather epistemic uncertainty
        for _ in range(self.num_of_infer):
            losses, preds, targets, std = self.model_step(copy.deepcopy(batch))
            
            # Append results to lists
            all_preds.append(preds)
            all_aleatoric_uncertainties.append(std)  # Aleatoric uncertainty (std from the model)

        # Stack the lists along a new dimension (assuming preds and targets are tensors with the same shape)
        all_preds = torch.stack(all_preds)
        all_aleatoric_uncertainties = torch.stack(all_aleatoric_uncertainties)

        # Epistemic uncertainty (variance of predictions)
        variance_preds = all_preds.var(dim=0)
        # Mean prediction
        preds = all_preds.mean(dim=0)
        # Mean aleatoric uncertainty
        aleatoric_uncertainty = all_aleatoric_uncertainties.mean(dim=0)

        # Save the results
        save_tensor_to_nifti(variance_preds, join(self.logger.save_dir, f"{batch['metadata']['File name'][0]}_preds_var.nii"))
        save_tensor_to_nifti(preds, join(self.logger.save_dir, f"{batch['metadata']['File name'][0]}_preds.nii"))
        save_tensor_to_nifti(targets, join(self.logger.save_dir, f"{batch['metadata']['File name'][0]}_targets.nii"))
        save_tensor_to_nifti(aleatoric_uncertainty, join(self.logger.save_dir, f"{batch['metadata']['File name'][0]}_aleatoric_uncertainty.nii"))
        save_tensor_to_nifti(targets - preds, join(self.logger.save_dir, f"{batch['metadata']['File name'][0]}_diff.nii"))

        accuracies = {}
        for key, acc in self.test_acc.items():
            acc_ = []
            for i in range(preds.shape[0]):
                acc(preds[i, None, None, ...], targets[i, None, None, ...])
                acc_.append(acc.compute())

            if key not in accuracies:
                accuracies[key] = []
            
            accuracies[key] = [(x.cpu().numpy()).tolist() for x in acc_]
            self.log(f"test_acc/{key}_mean", torch.mean(torch.Tensor(acc_)), on_step=True, on_epoch=True, prog_bar=False)
            self.log(f"test_acc/{key}_std", torch.std(torch.Tensor(acc_)), on_step=True, on_epoch=True, prog_bar=False)
            acc.reset()

        # Log test losses
        for key, loss in losses.items():
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
