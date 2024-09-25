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
        # Add noise to both real and imaginary parts (last dimension contains real and imaginary parts)
        noise = torch.randn_like(k_space_data_real) * self.noise_level

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
        # Separate real and imaginary parts
        noise_real = torch.randn_like(k_space_data.real) * self.noise_level
        noise_imag = torch.randn_like(k_space_data.imag) * self.noise_level

        # Add noise to real and imaginary parts
        noisy_k_space_data = k_space_data + torch.complex(noise_real, noise_imag)

        return noisy_k_space_data
