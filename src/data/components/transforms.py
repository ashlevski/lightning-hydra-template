import torch
from torchvision import transforms

class ComplexToReal(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return torch.view_as_real(tensor)
