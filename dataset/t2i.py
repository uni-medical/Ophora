import os
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import accimage_loader, pil_loader
from .builder import DATASETS
import mmengine
from mmengine.fileio import FileClient
import io
import torch.nn.functional as F
import random


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

@DATASETS.register_module()
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, data_loader=default_loader, return_name=False, image_suffix=(".jpg", ".jpeg", ".png")):
        self.transform = transform
        self.data_loader = data_loader
        self.return_name = return_name
        self.image_dir = image_dir
        self.image_list = [item for item in mmengine.scandir(image_dir, suffix=image_suffix, recursive=True)]


    def split_dataset(self, start_index, end_index):
        self.image_list = self.image_list[start_index:end_index]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = self.image_list[idx]
                img_dir = os.path.join(self.image_dir, img_name)
                img = self.data_loader(img_dir)
                if self.transform:
                    img = self.transform(img)
                if self.return_name:
                    return img, img_name
            except Exception as e:
                print(f"Error loading image at index {idx}: {str(e)}")
                idx = self.get_next_valid_index(idx)

    def get_next_valid_index(self, current_idx):
        next_idx = current_idx + 1
        if next_idx >= len(self.image_list):
            next_idx = 0  # Wrap around to the beginning of the dataset if necessary
        return next_idx

@DATASETS.register_module()
class TextDataset(Dataset):
    def __init__(self, text_data, transform=None, name_key="name", text_key=None, return_name=True):
        self.transform = transform
        self.return_name = return_name
        self.text_list = []

        if isinstance(text_data, list):
            self.text_list = text_data
            self.name_list = None
        elif isinstance(text_data, str):
            if text_data.endswith('.csv'):
                df = pd.read_csv(text_data)
                if text_key and text_key in df.columns:
                    self.text_list = df[text_key].tolist()
                else:
                    raise ValueError("text_key not found in CSV columns")
                if name_key and name_key in df.columns:
                    self.name_list = df[name_key].tolist()
            else:
                # Handle text file
                with open(text_data, 'r') as file:
                    self.text_list = file.read().splitlines()
                self.name_list = None

        self.text_list = self.text_list
        non_nan_indices = [i for i, text in enumerate(self.text_list) if not pd.isna(text)]
        self.text_list = [self.text_list[i] for i in non_nan_indices]
        if self.name_list:
            self.name_list = [self.name_list[i] for i in non_nan_indices]
        else:
            self.name_list = [str(i) for i in non_nan_indices]

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        if self.transform:
            text = self.transform(text)
        if self.return_name:
            if self.name_list is not None:
                return self.name_list[idx], text
            else:
                return str(idx), text
        return text
    
@DATASETS.register_module()
class T2IDataset(Dataset):
    def __init__(
        self,
        transform=None,
        data_loader=default_loader,
        img_embed_cache_dir=None,
        text_embed_cache_dir=None,
        cond_embed_cache_dir=None,
        data_list_file=None,
        data_list_split_delimiter=" ",
        return_name=False,
        use_cond=False,
        suffix=None,
        file_client_args=None,
        s3_bucket=None,
        random_start_frame=True,
        frame_stride_range=[1,8],
    ):
        self.transform = transform
        self.data_loader = data_loader
        self.use_cond = use_cond
        self.suffix = suffix
        self.file_client = FileClient(**(file_client_args or {}))
        self.s3_bucket = s3_bucket

        self.frame_stride_range = frame_stride_range
        self.random_start_frame = random_start_frame

        
        if data_list_file:
            self.data_list = self.load_data_list_from_file(data_list_file, data_list_split_delimiter)
        else:
            self.img_embed_cache_dir = img_embed_cache_dir
            self.text_embed_cache_dir = text_embed_cache_dir
            self.cond_embed_cache_dir = cond_embed_cache_dir if use_cond else None
            self.data_list = self.load_data_list()

        self.return_name = return_name

    def load_data_list(self):
        data_list = []
        image_names = self.file_client.list_dir_or_file(self.img_embed_cache_dir, suffix=self.suffix, recursive=True)
        for img_name in image_names:
            img_path = os.path.join(self.img_embed_cache_dir, img_name)
            text_path = os.path.join(self.text_embed_cache_dir, img_name)
            cond_path = os.path.join(self.cond_embed_cache_dir, img_name) if self.use_cond else None
            if self.file_client.isfile(text_path) and (not self.use_cond or self.file_client.isfile(cond_path)):
                entry = [img_path, text_path]
                if self.use_cond:
                    entry.append(cond_path)
                data_list.append(entry)
        return data_list

    def load_data_list_from_file(self, data_list_file, data_list_split_delimiter):
        data_list = []
        with open(data_list_file, 'r') as f:
            for line in f:
                paths = line.strip().split(data_list_split_delimiter)
                if self.s3_bucket is not None:
                    paths = [os.path.join(self.s3_bucket, path) for path in paths]
                if len(paths) == 2 or (len(paths) == 3 and self.use_cond):
                    data_list.append(paths)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        while True:
            try:
                data_entry = self.data_list[idx]
                image_file, text_file = data_entry[:2]
                cond_file = data_entry[2] if self.use_cond and len(data_entry) > 2 else None
                # Process image
                image_bytes = self.file_client.get(image_file)
                with io.BytesIO(image_bytes) as image_buffer:
                    image_data = np.load(image_buffer, allow_pickle=True)
                    image = torch.from_numpy(image_data['image']).float()
                if self.transform:
                    image = self.transform(image)

                # TODO remove in the future
                if len(image.shape) == 5:
                    image = image.squeeze(0)

                # Process text
                text_bytes = self.file_client.get(text_file)
                with io.BytesIO(text_bytes) as text_buffer:
                    text_data = np.load(text_buffer, allow_pickle=True)
                    text = torch.from_numpy(text_data['caption_embedding']).float()
                    text_mask = torch.from_numpy(text_data['caption_mask']).bool()

                return_data = [image, (text, text_mask)]

                if self.return_name:
                    name = os.path.basename(image_file).replace('.npz', '')
                    return_data.insert(0, name)

                if self.use_cond:
                    cond_bytes = self.file_client.get(cond_file)
                    with io.BytesIO(cond_bytes) as cond_buffer:
                        cond_data = np.load(cond_buffer, allow_pickle=True)
                        cond = torch.from_numpy(cond_data['image']).float()
                    return_data.append(cond)
                return return_data
            except Exception as e:
                    print(f"Error loading data at index {idx}: {str(e)}")
                    idx = self.get_next_valid_index(idx)

    def get_next_valid_index(self, current_idx):
        next_idx = current_idx + 1
        if next_idx >= len(self.data_list):
            next_idx = 0  # Optionally handle wrap-around to the start
        return next_idx
