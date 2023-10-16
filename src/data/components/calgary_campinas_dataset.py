import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from os import path

class SliceDataset(Dataset):
    def __init__(self, base_directory, data_df, split, smaps, us_masks, target_type, coils, data_transforms=None, target_transforms=None) -> None:
        super().__init__()
        '''
        Dataset class for 2D slices 
        Args:
            data_df (DataFrame): Contains slice paths, patient ids, slice numbers, and splits
            split (str): One of train, val, test
            smaps (List): Contains paths to the various sensitivity maps
            us_masks (List): Contains paths to the various undersampling masks
            target_type (str): ESPiRIT or NLINV used for recosntructing the target
            coils (int): how many coils in this multi-coil image
            data_transforms(callable, optional): Optional composition of tranforms for the input data
            target_transforms(callable, optional): Optional composition of transforms for the target data
        '''
        self.base_directory = base_directory
        self.file_paths = data_df.loc[data_df['split']==split]
        self.smaps = smaps
        self.us_masks = us_masks
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.coils = coils

    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, idx):
        img_path = self.file_paths.iloc[idx]
        img_path_ = path.join(self.base_directory, str(img_path['split']), 'target_slice_images',str(img_path['patient_id'])+'_s'+str(img_path['slice'])+'_nlinv_img.npy')
        # recall this is the nicely done reconstruction
        smap_path = random.choice(self.smaps) 
        us_mask_path = random.choice(self.us_masks)

        target_img = np.load(img_path_)  # has size 1, 218, 170. 1 channel image, dims 218 x 170.
        if target_img.shape[-1] != 170:
            diff = int((target_img.shape[-1] - 170) / 2)  # difference per side
            target_img = target_img[:, :, diff:-diff]
        smap = np.load(smap_path)  # size channels, h, w 
        mask = np.load(us_mask_path)
        mask = np.repeat(mask[None, :, :], self.coils, axis=0)

        noise = np.random.normal(0, 2/1000, target_img.shape) + 1j * np.random.normal(0, 2/1000, target_img.shape)
        input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap + noise, axes=(-1, -2))), axes=(-1, -2)) * mask
        input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2))   # want shape channel h w

        input_img =  np.moveaxis(input_img, 0, -1) # In numpy, want channels at end. Torch tensor transform will move them to the front
        target_img = np.moveaxis(target_img, 0, -1)
        smap = np.moveaxis(smap, 0, -1)
        input_kspace = torch.from_numpy(input_kspace)
        mask = torch.from_numpy(mask)

        if self.data_transforms:  # realistically, just ToTensor and then ReImgChannels
            input_img = self.data_transforms(input_img)
        if self.target_transforms:
            target_img = self.target_transforms(target_img)
            smap = self.target_transforms(smap)
        
        # scale by dividing all elements by the max value
        if input_img.dtype == torch.cdouble:
            input_max = torch.max(torch.abs(torch.view_as_real(input_img)))
        else:
            input_max = torch.max(torch.abs(input_img))

        input_img = torch.div(input_img, input_max)
        target_img = torch.div(target_img, input_max)
        input_kspace = torch.div(input_kspace, input_max)
        input_max = torch.reshape(input_max, (1, 1, 1))

        return input_img, target_img, smap, input_max, input_kspace, mask


class SliceTestDataset(Dataset):
    def __init__(self, data_df, split, smaps, us_masks, target_type, channels, data_transforms=None, target_transforms=None, rand='smap', smap_choice=None, mask_choice=None) -> None:
        super().__init__()
        '''
        Dataset class for 2D test slices. Only difference is now we can choose to set a certain smap or us mask. Also, we find out afterwards
        what was chosen
        Args:
            data_df (DataFrame): Contains slice paths, patient ids, slice numbers, and splits
            split (str): One of train, val, test
            smaps (List): Contains paths to the various sensitivity maps
            us_masks (List): Contains paths to the various undersampling masks
            target_type (str): ESPiRIT or NLINV used for recosntructing the target
            data_transforms(callable, optional): Optional composition of tranforms for the input data
            target_transforms(callable, optional): Optional composition of transforms for the target data
        '''
        if target_type == 'nlinv':
            self.file_paths = data_df.loc[data_df['split']==split, 'nlinv_path'].tolist() 
        else:
            self.file_paths = data_df.loc[data_df['split']==split, 'espirit_path'].tolist()

        self.smaps = smaps
        self.us_masks = us_masks
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.channels = channels

        self.rand = rand
        self.smap_choice = smap_choice
        self.mask_choice = mask_choice

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        fname, dataslice, metadata = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

        return sample
        

