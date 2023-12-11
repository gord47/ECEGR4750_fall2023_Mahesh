import torch
import math
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random

from PIL import Image
class CustomDataloader():
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, randomize: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)
        

    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):
        """
        This function randomizes the dataset, while maintaining the relationship between 
        x and y tensors
        """
        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'x_batch':self.x[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'y_batch':self.y[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
    
    def fetch_batch(self):
        """
        This function calls next on the batch iterator, and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        
        # if the iter hasn't been generated yet
        if self.iter == None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)
        batch_img_names = batch['x_batch']
        # detect if this is the final batch to avoid StopIteration error
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()
        return batch


class MultimodalDataset():
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, randomize: bool = False, image_folder: str = ""):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])

    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):
        """
        This function randomizes the dataset, while maintaining the relationship between 
        x and y tensors
        """
        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'x_batch':self.x[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'y_batch':self.y[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
        
    def fetch_batch(self):
        """
        This function calls next on the batch iterator, and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        
        # if the iter hasn't been generated yet
        if self.iter == None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)
        lower_df = batch['x_batch']
        img_names = lower_df['filename']
        
        images = [self.load_image(str(image_name)) for image_name in img_names]
        batch['images'] = torch.stack(images)
        batch['x_batch'] = batch['x_batch'].drop('filename', axis=1)
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()
        return batch
    
    def load_image(self, img_name):
        image_path = os.path.join(self.image_folder, img_name)
        image = Image.open(image_path).convert('RGB')
        if(self.transform):
            image = self.transform(image)
        return image