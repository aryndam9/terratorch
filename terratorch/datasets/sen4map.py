import numpy as np
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from terratorch.datasets.utils import HLSBands

from torchvision.transforms.v2.functional import resize
from torchvision.transforms.v2 import InterpolationMode

import pickle


class Sen4MapDataset(Dataset):
    """[Sen4Map](https://gitlab.jsc.fz-juelich.de/sdlrs/sen4map-benchmark-dataset) Dataset.

    Dataset intended for land-cover and crop classification tasks based on 
    satellite data stored in HDF5 files.

    Dataset Format:

    * HDF5 files containing spectral bands (e.g., B2, B3, …, B12)
    * Classification labels provided via HDF5 attributes (e.g., 'lc1') with mappings defined for:
        - Land-cover: using `land_cover_classification_map`
        - Crops: using `crop_classification_map`

    Dataset Features:

    * Supports two classification tasks: "land-cover" (default) and "crops".
    * Pre-processing options include center cropping and resizing.
    * Option to save the keys HDF5 for later filtering.
    * Input channel selection via a mapping between available bands and input bands.
    """
    land_cover_classification_map={'A10':0, 'A11':0, 'A12':0, 'A13':0, 
    'A20':0, 'A21':0, 'A30':0, 
    'A22':1, 'F10':1, 'F20':1, 
    'F30':1, 'F40':1,
    'E10':2, 'E20':2, 'E30':2, 'B50':2, 'B51':2, 'B52':2,
    'B53':2, 'B54':2, 'B55':2,
    'B10':3, 'B11':3, 'B12':3, 'B13':3, 'B14':3, 'B15':3,
    'B16':3, 'B17':3, 'B18':3, 'B19':3, 'B10':3, 'B20':3, 
    'B21':3, 'B22':3, 'B23':3, 'B30':3, 'B31':3, 'B32':3,
    'B33':3, 'B34':3, 'B35':3, 'B30':3, 'B36':3, 'B37':3,
    'B40':3, 'B41':3, 'B42':3, 'B43':3, 'B44':3, 'B45':3,
    'B70':3, 'B71':3, 'B72':3, 'B73':3, 'B74':3, 'B75':3,
    'B76':3, 'B77':3, 'B80':3, 'B81':3, 'B82':3, 'B83':3,
    'B84':3, 
    'BX1':3, 'BX2':3,
    'C10':4, 'C20':5, 'C21':5, 'C22':5,
    'C23':5, 'C30':5, 'C31':5, 'C32':5,
    'C33':5, 
    'CXX1':5, 'CXX2':5, 'CXX3':5, 'CXX4':5, 'CXX5':5,
    'CXX5':5, 'CXX6':5, 'CXX7':5, 'CXX8':5, 'CXX9':5,
    'CXXA':5, 'CXXB':5, 'CXXC':5, 'CXXD':5, 'CXXE':5,
    'D10':6, 'D20':6, 'D10':6,
    'G10':7, 'G11':7, 'G12':7, 'G20':7, 'G21':7, 'G22':7, 'G30':7, 
    'G40':7,
    'G50':7,
    'H10':8, 'H11':8, 'H12':8, 'H11':8,'H20':8, 'H21':8,
    'H22':8, 'H23':8, '': 9}
    #  This dictionary maps the LUCAS classes to crop classes.
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
            save_keys_path = None,
            classification_map = "land-cover"
            ):
        """Initialize a new instance of Sen4MapDataset.

        This dataset loads data from an HDF5 file object containing satellite data.

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
            save_keys_path: Optional file path to save the list of dataset keys.
            classification_map: String specifying the classification mapping to use ("land-cover" or "crops").
                Default is "land-cover".

        Raises:
            ValueError: If `input_bands` is provided without specifying `dataset_bands`.
            ValueError: If an invalid `classification_map` is provided.
        """
        self.h5data = h5py_file_object
        if h5data_keys is None:
            if classification_map == "crops": print(f"Crop classification task chosen but no keys supplied. Will fail unless dataset hdf5 files have been filtered. Either filter dataset files or create a filtered set of keys.")
            self.h5data_keys = list(self.h5data.keys())
            if save_keys_path is not None:
                with open(save_keys_path, "wb") as file:
                    pickle.dump(self.h5data_keys, file)
        else:
            self.h5data_keys = h5data_keys
        self.crop_size = crop_size
        if input_bands and not dataset_bands:
            raise ValueError(f"input_bands was provided without specifying the dataset_bands")
        # self.dataset_bands = dataset_bands
        # self.input_bands = input_bands
        if input_bands and dataset_bands:
            self.input_channels = [dataset_bands.index(band_ind) for band_ind in input_bands if band_ind in dataset_bands]
        else: self.input_channels = None

        classification_maps = {"land-cover": Sen4MapDataset.land_cover_classification_map,
                               "crops": Sen4MapDataset.crop_classification_map}
        if classification_map not in classification_maps.keys():
            raise ValueError(f"Provided classification_map of: {classification_map}, is not from the list of valid ones: {classification_maps}")
        self.classification_map = classification_maps[classification_map]

        self.resize = resize
        self.resize_to = resize_to
        self.resize_interpolation = resize_interpolation
        self.resize_antialiasing = resize_antialiasing

    def __getitem__(self, index):
        # we can call dataset with an index, eg. dataset[0]
        im = self.h5data[self.h5data_keys[index]]
        Image, Label = self.get_data(im)
        Image = self.min_max_normalize(Image, [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27],
                                    [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45])
        
        Image = Image.clip(0,1)
        Label = torch.LongTensor(Label)
        if self.input_channels:
            Image = Image[self.input_channels, ...]

        return {"image":Image, "label":Label}

    def __len__(self):
        return len(self.h5data_keys)

    def get_data(self, im):
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
        
        Image = np.stack((B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12), axis=0, dtype="float32")
        Image = np.moveaxis(Image, [0],[1])
        Image = torch.from_numpy(Image)
        
        if self.crop_size: 
            Image = self.crop_center(Image, self.crop_size, self.crop_size)
        
        if self.resize:
            Image = resize(Image, size=self.resize_to, interpolation=self.resize_interpolation, antialias=self.resize_antialiasing)

        Label = im.attrs['lc1']
        Label = self.classification_map[Label]
        Label = np.array(Label)
        Label = Label.astype('float32')

        return Image, Label
    
    def crop_center(self, img:torch.Tensor, cropx, cropy) -> torch.Tensor:
        c, y, x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[0:c, starty:starty+cropy, startx:startx+cropx]

    def min_max_normalize(self, tensor:torch.Tensor, q_low:list[float], q_hi:list[float]) -> torch.Tensor:
        dtype = tensor.dtype
        q_low = torch.as_tensor(q_low, dtype=dtype, device=tensor.device)
        q_hi = torch.as_tensor(q_hi, dtype=dtype, device=tensor.device)
        x = torch.tensor(-12.0)
        y = torch.exp(x)
        tensor.sub_(q_low[:, None, None]).div_((q_hi[:, None, None].sub_(q_low[:, None, None])).add(y))
        return tensor