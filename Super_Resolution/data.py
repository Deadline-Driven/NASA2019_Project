from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor, train_folder, train_ds_folder):
    crop_size = calculate_valid_crop_size(72, upscale_factor)

    return DatasetFromFolder(train_folder,
                             train_ds_folder,
                             input_transform=input_transform(crop_size),
                             target_transform=target_transform(crop_size*2))


def get_test_set(upscale_factor, test_folder, test_ds_folder):
    crop_size = calculate_valid_crop_size(72, upscale_factor)

    return DatasetFromFolder(test_folder,
                             test_ds_folder,
                             input_transform=input_transform(crop_size),
                             target_transform=target_transform(crop_size*2))
