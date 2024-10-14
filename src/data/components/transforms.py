from typing import Optional

import torch
from torchvision import transforms

class ComplexToReal(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return torch.view_as_real(tensor)


class KSpaceNoiseAdder_Real(object):
    def __init__(self, noise_level=0.01):
        """
        Initializes the noise adder with a given noise level.

        Args:
        noise_level (float): Standard deviation of the Gaussian noise to be added.
        """
        self.noise_level = noise_level

    def __call__(self, k_space_data_real):
        """
        Adds Gaussian noise to the real representation of the k-space data.

        Args:
        k_space_data_real (torch.Tensor): The k-space data in real form (shape: [..., 2], where the last dimension
                                          contains real and imaginary parts).

        Returns:
        torch.Tensor: The k-space data with added noise (in real form).
        """
        scale = torch.abs(k_space_data_real).max()
        # Add noise to both real and imaginary parts (last dimension contains real and imaginary parts)
        noise = torch.randn_like(k_space_data_real) * self.noise_level * scale

        # Adding noise to the real and imaginary components
        noisy_k_space_data_real = k_space_data_real + noise

        return noisy_k_space_data_real


import torch


class KSpaceNoiseAdder_Complex(object):
    def __init__(self, noise_level=0.01):
        """
        Initializes the noise adder with a given noise level.

        Args:
        noise_level (float): Standard deviation of the Gaussian noise to be added.
        """
        self.noise_level = noise_level

    def __call__(self, k_space_data):
        """
        Adds Gaussian noise to the k-space data.

        Args:
        k_space_data (torch.Tensor): The k-space data in complex form (torch.complex64 or torch.complex128).

        Returns:
        torch.Tensor: The k-space data with added noise.
        """
        scale = torch.abs(k_space_data).max()
        # Separate real and imaginary parts
        noise_real = torch.randn_like(k_space_data.real) * self.noise_level *scale
        noise_imag = torch.randn_like(k_space_data.imag) * self.noise_level *scale
        # Add noise to real and imaginary parts
        noisy_k_space_data = k_space_data + torch.complex(noise_real, noise_imag)

        return noisy_k_space_data


        # output_image = T.root_sum_of_squares(
        #     self.backward_operator(output_kspace, dim=self._spatial_dim),
        #     dim=self._coil_dim,
        # )  # shape (batch, height,  width)

import torchio as tio


class KSpaceSensitivityAndAffineTransform(object):
    def __init__(self,
                 scales: Optional[tuple] = None,
                 degrees: Optional[tuple] = None,
                 translation: Optional[tuple] = None
                 ):
        """
        Initializes the transform with optional affine parameters.

        Args:
        scales (tuple, optional): Scaling factors for the affine transformation.
        degrees (tuple, optional): Rotation angles (in degrees) for the affine transformation.
        translation (torch.Tensor, optional): 3D translation vector for affine transformation.
        """
        # If no scales or degrees are provided, default to identity transformation (no change)
        if scales is None:
            scales = (0,0,0.9,1.1,0.9,1.1)
        if degrees is None:
            degrees = (0,0,-45,45,-45,45)

        self.affine_transform = tio.transforms.RandomAffine(degrees=degrees)

    def calculate_sensitivity_maps(self, coil_images):
        """
        Calculate the sensitivity maps from coil images.

        Args:
        coil_images (torch.Tensor): Image domain data from the inverse FFT, shape [num_coils, height, width].

        Returns:
        torch.Tensor: Sensitivity maps of shape [num_coils, height, width].
        """
        # Sum the square of all coil images to calculate the root sum of squares (RSS)
        rss_image = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0, keepdim=True))

        # Calculate sensitivity map for each coil: coil_image / root sum of squares
        sensitivity_maps = coil_images / (rss_image + 1e-8)  # Add epsilon to avoid division by zero

        return sensitivity_maps

    def __call__(self, k_space_data):
        """
        Apply FFT, calculate sensitivity maps, affine transform in the image domain, and return to k-space.

        Args:
        k_space_data (torch.Tensor): Input k-space data (complex tensor) with shape [num_coils, height, width].

        Returns:
        torch.Tensor: Transformed k-space data (complex tensor).
        """
        num_coils = k_space_data.shape[0]

        # Convert k-space data to image domain using inverse FFT (ifft2)
        coil_images = torch.fft.ifft2(k_space_data)

        # Calculate sensitivity maps
        sensitivity_maps = self.calculate_sensitivity_maps(coil_images)

        # Compute the absolute image by combining all coils
        combined_image = torch.abs(torch.sum(coil_images * torch.conj(sensitivity_maps), dim=0))

        # Expand combined image to have a channel dimension (required by TorchIO)
        combined_image = combined_image.unsqueeze(0)  # Shape: [1, height, width]

        # Apply the affine transformation to the absolute image
        transformed_image = self.affine_transform(combined_image.unsqueeze(-1))  # Shape: [1, 1, height, width]

        # Remove batch dimension
        transformed_image = transformed_image.squeeze(0).squeeze(-1)

        # Transform the modified image back to k-space by applying FFT
        transformed_k_space = torch.fft.fft2(sensitivity_maps*transformed_image.squeeze())

        return transformed_k_space

class KSpaceSensitivityAndRandomElasticDeformation(object):
    def __init__(self,
                 num_control_points: Optional[tuple] = None,
                 locked_borders: Optional[int] = None,
                 ):
        """
        Initializes the transform with optional affine parameters.

        Args:
        scales (tuple, optional): Scaling factors for the affine transformation.
        degrees (tuple, optional): Rotation angles (in degrees) for the affine transformation.
        translation (torch.Tensor, optional): 3D translation vector for affine transformation.
        """
        # If no scales or degrees are provided, default to identity transformation (no change)
        if num_control_points is None:
            num_control_points = (9, 9, 9)
        if locked_borders is None:
            locked_borders = 2

        self.affine_transform = tio.transforms.RandomElasticDeformation(num_control_points=num_control_points,locked_borders=locked_borders)

    def calculate_sensitivity_maps(self, coil_images):
        """
        Calculate the sensitivity maps from coil images.

        Args:
        coil_images (torch.Tensor): Image domain data from the inverse FFT, shape [num_coils, height, width].

        Returns:
        torch.Tensor: Sensitivity maps of shape [num_coils, height, width].
        """
        # Sum the square of all coil images to calculate the root sum of squares (RSS)
        rss_image = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0, keepdim=True))

        # Calculate sensitivity map for each coil: coil_image / root sum of squares
        sensitivity_maps = coil_images / (rss_image + 1e-8)  # Add epsilon to avoid division by zero

        return sensitivity_maps

    def __call__(self, k_space_data):
        """
        Apply FFT, calculate sensitivity maps, affine transform in the image domain, and return to k-space.

        Args:
        k_space_data (torch.Tensor): Input k-space data (complex tensor) with shape [num_coils, height, width].

        Returns:
        torch.Tensor: Transformed k-space data (complex tensor).
        """
        num_coils = k_space_data.shape[0]

        # Convert k-space data to image domain using inverse FFT (ifft2)
        coil_images = torch.fft.ifft2(k_space_data)

        # Calculate sensitivity maps
        sensitivity_maps = self.calculate_sensitivity_maps(coil_images)

        # Compute the absolute image by combining all coils
        combined_image = torch.abs(torch.sum(coil_images * torch.conj(sensitivity_maps), dim=0))

        # Expand combined image to have a channel dimension (required by TorchIO)
        combined_image = combined_image.unsqueeze(0)  # Shape: [1, height, width]

        # Apply the affine transformation to the absolute image
        transformed_image = self.affine_transform(combined_image.unsqueeze(-1))  # Shape: [1, 1, height, width]

        # Remove batch dimension
        transformed_image = transformed_image.squeeze(0).squeeze(-1)

        # Transform the modified image back to k-space by applying FFT
        transformed_k_space = torch.fft.fft2(sensitivity_maps*transformed_image.squeeze())

        return transformed_k_space