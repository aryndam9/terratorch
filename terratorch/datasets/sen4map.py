import numpy as np
import h5py
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from terratorch.datasets.utils import HLSBands

from torchvision.transforms.v2.functional import resize
from torchvision.transforms.v2 import InterpolationMode

import pickle


# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sen4MapDatasetSimple(Dataset):
    """Modified Sen4Map Dataset that doesn't use monthly composites.
    
    This dataset directly uses the temporal data without monthly median aggregation.
    """
    land_cover_classification_map={
        'A10':0, 'A11':0, 'A12':0, 'A13':0, 
        'A20':0, 'A21':0, 'A30':0, 
        'A22':1, 'F10':1, 'F20':1, 
        'F30':1, 'F40':1,
        'E10':2, 'E20':2, 'E30':2, 'B50':2, 'B51':2, 'B52':2,
        'B53':2, 'B54':2, 'B55':2,
        'B10':3, 'B11':3, 'B12':3, 'B13':3, 'B14':3, 'B15':3,
        'B16':3, 'B17':3, 'B18':3, 'B19':3, 'B20':3, 
        'B21':3, 'B22':3, 'B23':3, 'B30':3, 'B31':3, 'B32':3,
        'B33':3, 'B34':3, 'B35':3, 'B36':3, 'B37':3,
        'B40':3, 'B41':3, 'B42':3, 'B43':3, 'B44':3, 'B45':3,
        'B70':3, 'B71':3, 'B72':3, 'B73':3, 'B74':3, 'B75':3,
        'B76':3, 'B77':3, 'B80':3, 'B81':3, 'B82':3, 'B83':3,
        'B84':3, 
        'BX1':3, 'BX2':3,
        'C10':4, 'C20':5, 'C21':5, 'C22':5,
        'C23':5, 'C30':5, 'C31':5, 'C32':5,
        'C33':5, 
        'CXX1':5, 'CXX2':5, 'CXX3':5, 'CXX4':5, 'CXX5':5,
        'CXX6':5, 'CXX7':5, 'CXX8':5, 'CXX9':5,
        'CXXA':5, 'CXXB':5, 'CXXC':5, 'CXXD':5, 'CXXE':5,
        'D10':6, 'D20':6,
        'G10':7, 'G11':7, 'G12':7, 'G20':7, 'G21':7, 'G22':7, 'G30':7, 
        'G40':7, 'G50':7,
        'H10':8, 'H11':8, 'H12':8, 'H20':8, 'H21':8,
        'H22':8, 'H23':8, '': 9
    }
    
    crop_classification_map = {
        "B11":0, "B12":0, "B13":0, "B14":0, "B15":0, "B16":0, "B17":0, "B18":0, "B19":0,  # Cereals
        "B21":1, "B22":1, "B23":1,  # Root Crops
        "B31":2, "B32":2, "B33":2, "B34":2, "B35":2, "B36":2, "B37":2,  # Nonpermanent Industrial Crops
        "B41":3, "B42":3, "B43":3, "B44":3, "B45":3,  # Dry Pulses, Vegetables and Flowers
        "B51":4, "B52":4, "B53":4, "B54":4,  # Fodder Crops
        "F10":5, "F20":5, "F30":5, "F40":5,  # Bareland
        "B71":6, "B72":6, "B73":6, "B74":6, "B75":6, "B76":6, "B77":6, 
        "B81":6, "B82":6, "B83":6, "B84":6, "C10":6, "C21":6, "C22":6, "C23":6, "C31":6, "C32":6, "C33":6, "D10":6, "D20":6,  # Woodland and Shrubland
        "B55":7, "E10":7, "E20":7, "E30":7,  # Grassland
    }
    
    def __init__(
            self,
            h5py_file_object:h5py.File,
            h5data_keys = None,
            crop_size:None|int = None,
            dataset_bands:list[HLSBands|int]|None = None,
            input_bands:list[HLSBands|int]|None = None,
            resize = False,
            resize_to = [224, 224],
            resize_interpolation = InterpolationMode.BILINEAR,
            resize_antialiasing = True,
            reverse_tile = False,
            reverse_tile_size = 3,
            save_keys_path = None,
            classification_map = "land-cover"
            ):
        """Initialize a new instance of Sen4MapDatasetSimple.

        This dataset loads data from an HDF5 file object containing temporal satellite data
        without monthly median compositing.

        Args:
            h5py_file_object: An open h5py.File object containing the dataset.
            h5data_keys: Optional list of keys to select a subset of data samples from the HDF5 file.
                If None, all keys are used.
            crop_size: Optional integer specifying the square crop size for the output image.
            dataset_bands: Optional list of bands available in the dataset.
            input_bands: Optional list of bands to be used as input channels.
                Must be provided along with `dataset_bands`.
            resize: Boolean flag indicating whether the image should be resized. Default is False.
            resize_to: Target dimensions [height, width] for resizing. Default is [224, 224].
            resize_interpolation: Interpolation mode used for resizing. Default is InterpolationMode.BILINEAR.
            resize_antialiasing: Boolean flag to apply antialiasing during resizing. Default is True.
            reverse_tile: Boolean flag indicating whether to apply reverse tiling to the image. Default is False.
            reverse_tile_size: Kernel size for the reverse tiling operation. Must be an odd number >= 3. Default is 3.
            save_keys_path: Optional file path to save the list of dataset keys.
            classification_map: String specifying the classification mapping to use ("land-cover" or "crops").
                Default is "land-cover".

        Raises:
            ValueError: If `input_bands` is provided without specifying `dataset_bands`.
            ValueError: If an invalid `classification_map` is provided.
        """
        self.h5data = h5py_file_object
        if h5data_keys is None:
            if classification_map == "crops":
                logger.warning("Crop classification task chosen but no keys supplied. Will fail unless dataset hdf5 files have been filtered. Either filter dataset files or create a filtered set of keys.", stacklevel=2)
            self.h5data_keys = list(self.h5data.keys())
            if save_keys_path is not None:
                with open(save_keys_path, "wb") as file:
                    pickle.dump(self.h5data_keys, file)
        else:
            self.h5data_keys = h5data_keys
        self.crop_size = crop_size
        if input_bands and not dataset_bands:
            raise ValueError("input_bands was provided without specifying the dataset_bands")
        
        if input_bands and dataset_bands:
            self.input_channels = [dataset_bands.index(band_ind) for band_ind in input_bands if band_ind in dataset_bands]
        else: 
            self.input_channels = None

        classification_maps = {"land-cover": Sen4MapDatasetSimple.land_cover_classification_map,
                               "crops": Sen4MapDatasetSimple.crop_classification_map}
        if classification_map not in classification_maps.keys():
            raise ValueError(f"Provided classification_map of: {classification_map}, is not from the list of valid ones: {classification_maps}")
        self.classification_map = classification_maps[classification_map]

        self.resize = resize
        self.resize_to = resize_to
        self.resize_interpolation = resize_interpolation
        self.resize_antialiasing = resize_antialiasing
        
        self.reverse_tile = reverse_tile
        self.reverse_tile_size = reverse_tile_size

    def __getitem__(self, index):
        # we can call dataset with an index, eg. dataset[0]
        im = self.h5data[self.h5data_keys[index]]
        Image, Label = self.get_data(im)
        Image = self.min_max_normalize(Image, [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27],
                                     [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45])
        
        Image = Image.clip(0, 1)
        Label = torch.LongTensor(Label)
        if self.input_channels:
            Image = Image[self.input_channels, ...]

        return {"image": Image, "label": Label}

    def __len__(self):
        return len(self.h5data_keys)

    def get_data(self, im):
        # Log the shape of the input data
        logger.info(f"Input data shape: {im.shape}", stacklevel=2)
        
        mask = im['SCL'] < 9

        B2 = np.where(mask==1, im['B2'], 0)
        B3 = np.where(mask==1, im['B3'], 0)
        B4 = np.where(mask==1, im['B4'], 0)
        B5 = np.where(mask==1, im['B5'], 0)
        B6 = np.where(mask==1, im['B6'], 0)
        B7 = np.where(mask==1, im['B7'], 0)
        B8 = np.where(mask==1, im['B8'], 0)
        B8A = np.where(mask==1, im['B8A'], 0)
        B11 = np.where(mask==1, im['B11'], 0)
        B12 = np.where(mask==1, im['B12'], 0)

        # Stack the bands along the first dimension
        Image = np.stack((B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12), axis=0, dtype="float32")
        logger.info(f"Stacked image shape: {Image.shape}", stacklevel=2)
        
        # Move the band dimension to position 1 (channels dimension)
        Image = np.moveaxis(Image, [0], [1])
        logger.info(f"After moveaxis: {Image.shape}", stacklevel=2)
        
        # Convert to torch tensor
        Image = torch.from_numpy(Image)
        logger.info(f"Torch tensor shape: {Image.shape}", stacklevel=2)
        
        # Apply center cropping if specified
        if self.crop_size: 
            Image = self.crop_center(Image, self.crop_size, self.crop_size)
            logger.info(f"After cropping: {Image.shape}", stacklevel=2)
        
        # Apply reverse tiling if specified
        if self.reverse_tile:
            Image = self.reverse_tiling_pytorch(Image, kernel_size=self.reverse_tile_size)
            logger.info(f"After reverse tiling: {Image.shape}", stacklevel=2)
        
        # Apply resizing if specified
        if self.resize:
            Image = resize(Image, size=self.resize_to, interpolation=self.resize_interpolation, antialias=self.resize_antialiasing)
            logger.info(f"After resizing: {Image.shape}", stacklevel=2)

        # Get the label
        Label = im.attrs['lc1']
        Label = self.classification_map[Label]
        Label = np.array(Label)
        Label = Label.astype('float32')
        logger.info(f"Label: {Label}", stacklevel=2)

        return Image, Label
    
    def crop_center(self, img_b:torch.Tensor, cropx, cropy) -> torch.Tensor:
        """Center crop the image tensor.
        
        Args:
            img_b: Input tensor with shape [channels, time, height, width]
            cropx: Width of the crop
            cropy: Height of the crop
            
        Returns:
            Center-cropped tensor
        """
        # Check the shape of the tensor
        if len(img_b.shape) == 4:
            c, t, y, x = img_b.shape
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)    
            return img_b[0:c, 0:t, starty:starty+cropy, startx:startx+cropx]
        elif len(img_b.shape) == 3:
            # Handle case where there's no time dimension
            c, y, x = img_b.shape
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)
            return img_b[0:c, starty:starty+cropy, startx:startx+cropx]
        else:
            logger.warning(f"Unexpected tensor shape for cropping: {img_b.shape}", stacklevel=2)
            return img_b
    
    def reverse_tiling_pytorch(self, img_tensor: torch.Tensor, kernel_size: int=3):
        """
        Upscales an image where every pixel is expanded into `kernel_size`*`kernel_size` pixels.
        Used to test whether the benefit of resizing images to the pre-trained size comes from the bilnearly interpolated pixels,
        or if the same would be realized with no interpolated pixels.
        """
        assert kernel_size % 2 == 1
        assert kernel_size >= 3
        padding = (kernel_size - 1) // 2
        
        # Log the input tensor shape
        logger.info(f"Input tensor shape for reverse tiling: {img_tensor.shape}", stacklevel=2)
        
        # Check tensor dimensions and adjust accordingly
        if len(img_tensor.shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, H, W = img_tensor.shape
        elif len(img_tensor.shape) == 3:  # [channels, height, width]
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            batch_size, channels, H, W = img_tensor.shape
        else:
            logger.warning(f"Unexpected tensor shape for reverse tiling: {img_tensor.shape}", stacklevel=2)
            return img_tensor
            
        # Unfold: Extract patches with padding to cover borders
        img_tensor = F.pad(img_tensor, pad=(padding, padding, padding, padding), mode="replicate")
        patches = F.unfold(img_tensor, kernel_size=kernel_size, padding=0)
        
        # Reshape to organize the values from each neighborhood
        patches = patches.view(batch_size, channels, kernel_size*kernel_size, H, W)
        
        # Rearrange the patches
        patches = patches.view(batch_size, channels, kernel_size, kernel_size, H, W)
        
        # Permute to have the spatial dimensions first and unfold them
        patches = patches.permute(0, 1, 4, 2, 5, 3)
        
        # Reshape to get the final expanded image
        expanded_img = patches.reshape(batch_size, channels, H * kernel_size, W * kernel_size)
        
        # Remove batch dimension if it wasn't in the input
        if len(img_tensor.shape) == 3:
            expanded_img = expanded_img.squeeze(0)
            
        logger.info(f"Output tensor shape after reverse tiling: {expanded_img.shape}", stacklevel=2)
        return expanded_img

    def min_max_normalize(self, tensor:torch.Tensor, q_low:list[float], q_hi:list[float]) -> torch.Tensor:
        """
        Normalize tensor values to range [0, 1] based on provided min/max quantiles.
        
        Args:
            tensor: Input tensor to normalize
            q_low: List of lower quantile values for each channel
            q_hi: List of upper quantile values for each channel
            
        Returns:
            Normalized tensor
        """
        # Log input tensor shape
        logger.info(f"Normalizing tensor of shape: {tensor.shape}", stacklevel=2)
        
        dtype = tensor.dtype
        q_low = torch.as_tensor(q_low, dtype=dtype, device=tensor.device)
        q_hi = torch.as_tensor(q_hi, dtype=dtype, device=tensor.device)
        
        # Add small epsilon to avoid division by zero
        x = torch.tensor(-12.0)
        y = torch.exp(x)
        
        # Adjust q_low shape based on tensor dimensions
        if len(tensor.shape) == 4:  # [channels, time, height, width]
            tensor.sub_(q_low[:, None, None, None]).div_((q_hi[:, None, None, None].sub_(q_low[:, None, None, None])).add(y))
        elif len(tensor.shape) == 3:  # [channels, height, width]
            tensor.sub_(q_low[:, None, None]).div_((q_hi[:, None, None].sub_(q_low[:, None, None])).add(y))
        else:
            logger.warning(f"Unexpected tensor shape for normalization: {tensor.shape}", stacklevel=2)
            
        return tensor