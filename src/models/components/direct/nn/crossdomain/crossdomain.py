# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional, Union

import torch
import torch.nn as nn

import direct.data.transforms as T


class CrossDomainNetwork(nn.Module):
    """This performs optimisation in both, k-space ("K") and image ("I") domains according to domain_sequence."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        image_model_list: nn.ModuleList,
        kspace_model_list: Optional[Union[nn.ModuleList, None]] = None,
        domain_sequence: str = "KIKI",
        image_buffer_size: int = 1,
        kspace_buffer_size: int = 1,
        normalize_image: bool = False,
        **kwargs,
    ):
        """Inits CrossDomainNetwork.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        image_model_list: nn.ModuleList
            Image domain model list.
        kspace_model_list: Optional[nn.ModuleList]
            K-space domain model list. If set to None, a correction step is applied. Default: None.
        domain_sequence: str
            Domain sequence containing only "K" (k-space domain) and/or "I" (image domain). Default: "KIKI".
        image_buffer_size: int
            Image buffer size. Default: 1.
        kspace_buffer_size: int
            K-space buffer size. Default: 1.
        normalize_image: bool
            If True, input is normalized. Default: False.
        kwargs: dict
            Keyword Arguments.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.domain_sequence = [domain_name for domain_name in domain_sequence.strip()]
        if not set(self.domain_sequence).issubset({"K", "I"}):
            raise ValueError(f"Invalid domain sequence. Got {domain_sequence}. Should only contain 'K' and 'I'.")

        if kspace_model_list is not None:
            if len(kspace_model_list) != self.domain_sequence.count("K"):
                raise ValueError("K-space domain steps do not match k-space model list length.")

        if len(image_model_list) != self.domain_sequence.count("I"):
            raise ValueError("Image domain steps do not match image model list length.")

        self.kspace_model_list = kspace_model_list
        self.kspace_buffer_size = kspace_buffer_size

        self.image_model_list = image_model_list
        self.image_buffer_size = image_buffer_size

        self.normalize_image = normalize_image

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def kspace_correction(
        self,
        block_idx: int,
        image_buffer: torch.Tensor,
        kspace_buffer: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        masked_kspace: torch.Tensor,
    ) -> torch.Tensor:
        forward_buffer = torch.cat(
            [
                self._forward_operator(
                    image.clone(),
                    sampling_mask,
                    sensitivity_map,
                )
                for image in torch.split(image_buffer, 2, self._complex_dim)
            ],
            self._complex_dim,
        )
        kspace_buffer = torch.cat([kspace_buffer, forward_buffer, masked_kspace], self._complex_dim)

        if self.kspace_model_list is not None:
            kspace_buffer = self.kspace_model_list[block_idx](kspace_buffer.permute(0, 1, 4, 2, 3)).permute(
                0, 1, 3, 4, 2
            )
        else:
            kspace_buffer = kspace_buffer[..., :2] - kspace_buffer[..., 2:4]

        return kspace_buffer

    def image_correction(
        self,
        block_idx: int,
        image_buffer: torch.Tensor,
        kspace_buffer: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        backward_buffer = torch.cat(
            [
                self._backward_operator(kspace.clone(), sampling_mask, sensitivity_map)
                for kspace in torch.split(kspace_buffer, 2, self._complex_dim)
            ],
            self._complex_dim,
        )

        image_buffer = torch.cat([image_buffer, backward_buffer], self._complex_dim).permute(0, 3, 1, 2)
        image_buffer = self.image_model_list[block_idx](image_buffer).permute(0, 2, 3, 1)

        return image_buffer

    def _forward_operator(
        self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
        )
        return forward

    def _backward_operator(
        self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        backward = T.reduce_operator(
            self.backward_operator(
                torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                    kspace,
                ).contiguous(),
                self._spatial_dims,
            ),
            sensitivity_map,
            self._coil_dim,
        )
        return backward

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        scaling_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the forward pass of :class:`CrossDomainNetwork`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        scaling_factor: Optional[torch.Tensor]
            Scaling factor of shape (N,). If None, no scaling is applied. Default: None.

        Returns
        -------
        out_image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)

        if self.normalize_image and scaling_factor is not None:
            input_image = input_image / scaling_factor**2
            masked_kspace = masked_kspace / scaling_factor**2

        image_buffer = torch.cat([input_image] * self.image_buffer_size, self._complex_dim).to(masked_kspace.device)

        kspace_buffer = torch.cat([masked_kspace] * self.kspace_buffer_size, self._complex_dim).to(
            masked_kspace.device
        )

        kspace_block_idx, image_block_idx = 0, 0
        for block_domain in self.domain_sequence:
            if block_domain == "K":
                kspace_buffer = self.kspace_correction(
                    kspace_block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map, masked_kspace
                )
                kspace_block_idx += 1
            else:
                image_buffer = self.image_correction(
                    image_block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map
                )
                image_block_idx += 1

        if self.normalize_image and scaling_factor is not None:
            image_buffer = image_buffer * scaling_factor**2

        out_image = image_buffer[..., :2]
        return out_image
