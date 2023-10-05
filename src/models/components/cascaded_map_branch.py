import torch
from torch.nn import Conv2d, LeakyReLU, Sequential, Module


def to_re_img(batch):   # changes complex into re/img channels
    b, c, h, w = batch.shape
    empty_batch = torch.empty((b, c * 2, h , w))
    re_img, im_img = torch.real(batch), torch.imag(batch)
    empty_batch[:, ::2, :, :] = re_img
    empty_batch[:, 1::2, :, :] = im_img

    return empty_batch.to('cuda:0')

def to_complex(batch): # changes re/img channels into complex valued data
    batch = batch[:, ::2, :, :] + 1j * batch[:, 1::2, :, :]

    return batch


class DataConsistencyLayer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, og_kspace, mask):
        x = to_complex(x)
        mod_x_kspace = og_kspace + torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1)) * (1.0 - mask)  #x is standard image
        mod_x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(mod_x_kspace, dim=(-2, -1))), dim=(-2, -1))  
        mod_x = to_re_img(mod_x) 

        return mod_x


class CascadeBlock(Module):
    def __init__(self, input_channels, block_depth, filters) -> None:
        super().__init__()

        layers = []
        layers.append(Conv2d(input_channels, filters, kernel_size=3, padding='same'))
        layers.append(LeakyReLU())
        for i in range(block_depth - 2):
            layers.append(Conv2d(filters, filters, kernel_size=3, padding='same'))
            layers.append(LeakyReLU())
        layers.append(Conv2d(filters, input_channels, kernel_size=3, padding='same'))
        layers.append(LeakyReLU())

        self.layers = Sequential(*layers)
        self.dc = DataConsistencyLayer()

    def forward(self, info_tuple) -> torch.Tensor:
        x, og_kspace, mask = info_tuple
        x = self.layers(x)
        x = self.dc(x, og_kspace, mask)

        return (x, og_kspace, mask)
    

class FinalBlock(Module):
    def __init__(self, input_channels, block_depth, filters) -> None:
        super().__init__()

        layers = []
        layers.append(Conv2d(input_channels * 2, filters, kernel_size=3, padding='same'))
        layers.append(LeakyReLU())
        for i in range(block_depth - 2):
            layers.append(Conv2d(filters, filters, kernel_size=3, padding='same'))
            layers.append(LeakyReLU())
        layers.append(Conv2d(filters, 2, kernel_size=3, padding='same'))

        self.layers = Sequential(*layers)

    def forward(self, x_smap) -> torch.Tensor:
        x = self.layers(x_smap)

        return x


class SmapBlock(Module):
    def __init__(self, input_channels, smap_layers, smap_filters) -> None:
        super().__init__()
        layers = []
        layers.append(Conv2d(input_channels, smap_filters, kernel_size=3, padding='same'))
        for i in range(smap_layers - 2):
            layers.append(Conv2d(smap_filters, smap_filters, kernel_size=3, padding='same'))
            layers.append(LeakyReLU())
        layers.append(Conv2d(smap_filters, input_channels, kernel_size=3, padding='same'))
        self.layers = Sequential(*layers)
    def forward(self, info_tuple) -> torch.Tensor:
        x = info_tuple[0]
        smap = self.layers(x)

        return smap


class CascadedModel(Module):
    def __init__(self, input_channels, reps=5, block_depth=5, filters=110, smap_layers=8, smap_filters=110) -> None:
        super().__init__()
        # can iterate through a block or go through layers
        self.reps = reps
        self.final_block = FinalBlock(input_channels, block_depth, filters)
        self.smap_block = SmapBlock(input_channels, smap_layers, smap_filters)

        blocks = []
        for i in range(self.reps):
            blocks.append(CascadeBlock(input_channels, block_depth, filters))
        self.blocks = Sequential(*blocks)

    # batch is batchsize, channel, h, w
    def forward(self, info_tuple):  # info tuple is img input, kspace, mask
        x = self.blocks(info_tuple)[0]
        smap = self.smap_block(info_tuple)
        x_smap = torch.concat([x, smap], dim=1)  # first channels are for images, last for smaps
        x = self.final_block(x_smap)

        return x, smap
    

def test(): 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    # B, C, H, W
    input_img = torch.randn(size=(1, 8, 256, 256), dtype=torch.float32).to(device)  # size 1, 2, 256, 256
    ref_kspace = torch.complex(torch.randn(size=(1, 4, 256, 256)), torch.randn(size=(1, 4, 256, 256))).to(device)
    mask = torch.randn(size=(1, 4, 256, 256), dtype=torch.float32).to(device)

    input_channels = 8
    model = CascadedModel(input_channels).type(torch.float32).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    alt_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {pytorch_total_params}")

    input_tuple = (input_img, ref_kspace, mask)
    img_output, smap_output = model(input_tuple)
    print(input_img.shape, img_output.shape, smap_output.shape)

if __name__ == "__main__":
    test()