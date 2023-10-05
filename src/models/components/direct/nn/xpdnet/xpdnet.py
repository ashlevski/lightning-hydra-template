# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional

import torch.nn as nn

from direct.nn.conv.conv import Conv2d
from direct.nn.crossdomain.crossdomain import CrossDomainNetwork
from direct.nn.crossdomain.multicoil import MultiCoil
from direct.nn.didn.didn import DIDN
from direct.nn.mwcnn.mwcnn import MWCNN


class XPDNet(CrossDomainNetwork):
    """XPDNet as implemented in [1]_.

    References
    ----------

    .. [1] Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI Challenge.” ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_primal: int = 5,
        num_dual: int = 1,
        num_iter: int = 10,
        use_primal_only: bool = True,
        image_model_architecture: str = "MWCNN",
        kspace_model_architecture: Optional[str] = None,
        normalize: bool = False,
        **kwargs,
    ):
        """Inits :class:`XPDNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_primal: int
            Number of primal networks.
        num_dual: int
            Number of dual networks.
        num_iter: int
            Number of unrolled iterations.
        use_primal_only: bool
            If set to True no dual-kspace model is used. Default: True.
        image_model_architecture: str
            Primal-image model architecture. Currently only implemented for MWCNN. Default: 'MWCNN'.
        kspace_model_architecture: str
            Dual-kspace model architecture. Currently only implemented for CONV and DIDN.
        normalize: bool
            Normalize input. Default: False.
        kwargs: dict
            Keyword arguments for model architectures.
        """
        if use_primal_only:
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == "CONV":
            kspace_model_list = nn.ModuleList(
                [
                    MultiCoil(
                        Conv2d(
                            2 * (num_dual + num_primal + 1),
                            2 * num_dual,
                            kwargs.get("dual_conv_hidden_channels", 16),
                            kwargs.get("dual_conv_n_convs", 4),
                            batchnorm=kwargs.get("dual_conv_batchnorm", False),
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model_list = nn.ModuleList(
                [
                    MultiCoil(
                        DIDN(
                            in_channels=2 * (num_dual + num_primal + 1),
                            out_channels=2 * num_dual,
                            hidden_channels=kwargs.get("dual_didn_hidden_channels", 16),
                            num_dubs=kwargs.get("dual_didn_num_dubs", 6),
                            num_convs_recon=kwargs.get("dual_didn_num_convs_recon", 9),
                        )
                    )
                    for _ in range(num_iter)
                ]
            )

        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )
        if image_model_architecture == "MWCNN":
            image_model_list = nn.ModuleList(
                [
                    nn.Sequential(
                        MWCNN(
                            input_channels=2 * (num_primal + num_dual),
                            first_conv_hidden_channels=kwargs.get("mwcnn_hidden_channels", 32),
                            num_scales=kwargs.get("mwcnn_num_scales", 4),
                            bias=kwargs.get("mwcnn_bias", False),
                            batchnorm=kwargs.get("mwcnn_batchnorm", False),
                        ),
                        nn.Conv2d(2 * (num_primal + num_dual), 2 * num_primal, kernel_size=3, padding=1),
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented only with image_model_architecture == 'MWCNN'."
                f"Got {image_model_architecture}."
            )
        super().__init__(
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            image_model_list=image_model_list,
            kspace_model_list=kspace_model_list,
            domain_sequence="KI" * num_iter,
            image_buffer_size=num_primal,
            kspace_buffer_size=num_dual,
            normalize_image=normalize,
        )
