from __future__ import print_function, division
import os
import torch
#import argparse
import random
import json
#import math
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
from PIL import Image
import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms



np.random.seed(148)
'''
################################################################################
MIDAIR DATASET CLASS
    takes the transforms, path to the data, and path to the annotation JSON /
    manifest file and gives you bounding box info in a numpy array:
    [   class_ID        0 for posts, 1 for signs
        width           width of bounding box
        top             coordinate of topmost pixels
        height          height of bounding box
        left            coordinate of leftmost pixels
    ]

    __init__            creates the dataset object
    __len__             gives you the length of the dataset
    __getitem__         allows you to index into the dataset
                        e.g.    if the dataset is called "dataset", dataset[i]
                                gives you the dictionary for the i'th piece of
                                data with keys 'image' and 'boxes', where
                                'image' is the loaded image from the file and
                                'boxes' is a numpy array containing bounding
                                boxes as described above.
################################################################################
'''
class midairDataset(Dataset):

    def __init__(self, main_path, data_path, transform=None):
        # parse JSON file
        self.data = [json.loads(line) for line in open(os.path.join(main_path,
            "annotations\\signs-and-posts\\manifests\\output\\output.manifest"),'r')]
        self.data = pd.DataFrame.from_dict(self.data)
        self.transform = transform
        self.data_path = data_path

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        s3_path = self.data.iloc[idx,0]
        img_name = os.path.join(self.data_path, s3_path[27:])
        image = io.imread(img_name)
        boxes = np.asarray([])

        if not ((self.data.iloc[idx]).isnull().values.any()):
            boxes_dict = self.data.iloc[idx,1]
            boxes_dict = boxes_dict["annotations"]
            boxes = []
            for dicty in boxes_dict:
                boxes.append([dicty['class_id'],dicty['width'],dicty['top'],dicty['height'],dicty['left'],])
            boxes = np.array(boxes)
            boxes = boxes.astype('float').reshape(-1,5)
        sample = (image, boxes)

        if self.transform:
            sample = self.transform(sample)

        return sample

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

class midairDatasetYoloBL(Dataset):
    def __init__(self, main_path, data_path, img_size=416, augment=False, multiscale=False, normalized_labels=False):
        # parse JSON file
        self.data = [json.loads(line) for line in open(os.path.join(main_path,
            "annotations\\signs-and-posts\\manifests\\output\\output.manifest"),'r')]
        self.data = pd.DataFrame.from_dict(self.data)

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        
        self.data_path = data_path
        self.default_img_size = 1024

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s3_path = self.data.iloc[idx,0]
        #---------
        #  Image - Scale and pad
        #---------
        img_path = os.path.join(self.data_path, s3_path[27:])
        
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        #---------
        #  Label - Extract and convert to padded format
        #---------
        targets = None
        if not ((self.data.iloc[idx]).isnull().values.any()):
            boxes_dict = self.data.iloc[idx,1]
            boxes_dict = boxes_dict["annotations"]
            boxes = []
            # create a labels matrix
            for dicty in boxes_dict:
                boxes.append([dicty['class_id'], dicty['left'], dicty['top'], dicty['width'],dicty['height']]) # These are unscaled
            boxes = np.array(boxes)
            boxes = torch.from_numpy(boxes.astype('float').reshape(-1,5))
            
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1])
            y1 = h_factor * (boxes[:, 2])
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3])
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4])
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
                
        return img_path, img, targets


    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets


'''#############################################################################

             ⢠⠣⡑⡕⡱⡸⡀⡢⡂⢨⠀⡌⠀⠀⠀⠀⠀⠀           ░░░░▄███▓███████▓▓▓░░░░
 ⠀⠀⠀⠀⠀⠀⠀⡕⢅⠕⢘⢜⠰⣱⢱⢱⢕⢵⠰⡱⡱⢘⡄⡎⠌⡀⠀⠀⠀⠀       ░░░███░░░▒▒▒██████▓▓░░░
 ⠀⠀⠀⠀⠀ ⠀⠱⡸⡸⡨⢸⢸⢈⢮⡪⣣⣣⡣⡇⣫⡺⡸⡜⡎⡢⠀⠀⠀⠀⠀      ░░██░░░░░░▒▒▒██████▓▓░░
 ⠀⠀⠀⠀⠀ ⢱⢱⠵⢹⢸⢼⡐⡵⣝⢮⢖⢯⡪⡲⡝⠕⣝⢮⢪⢀⠀⠀⠀⠀ ⠀     ░██▄▄▄▄░░░▄▄▄▄█████▓▓░░
 ⠀⠀⠀ ⢀⠂⡮⠁⠐⠀⡀⡀⠑⢝⢮⣳⣫⢳⡙⠐⠀⡠⡀⠀⠑⠀⠀⠀⠀⠀        ░██░(◐)░░░▒(◐)▒█████
 ⠀⠀⠀⠀⢠⠣⠐⠀ ⭕ ￼ ⠀⠀⢪⢺⣪⢣⠀⡀ ⭕ .⠈⡈⠀⡀⠀⠀ ⠀     ░██░(◐)░░░▒(◐)▒██████
 ⠀⠀ ⠀⠐⡝⣕⢄⡀⠑⢙⠉⠁⡠⡣⢯⡪⣇⢇⢀⠀⠡⠁⠁⡠⡢⠡⠀⠀⠀ ⠀    ░██░(◐)░░░▒(◐)▒██████
 ⠀ ⠀⠀⠀⢑⢕⢧⣣⢐⡄⣄⡍⡎⡮⣳⢽⡸⡸⡊⣧⣢⠀⣕⠜⡌⠌⠀⠀⠀ ⠀    ░██░░░░░░░▒▒▒▒▒█████▓▓░
 ⠀ ⠀⠀⠀⠀⠌⡪⡪⠳⣝⢞⡆⡇⡣⡯⣞⢜⡜⡄⡧⡗⡇⠣⡃⡂⠀⠀⠀⠀ ⠀⠀   ░██░░░▀▄▄▀▒▒▒▒▒█████▓▓░
   ⠀⠀⠀⠀⠀⠨⢊⢜⢜⣝⣪⢪⠌⢩⢪⢃⢱⣱⢹⢪⢪⠊⠀⠀⠀⠀⠀⠀ ⠀⠀⠀   ░█░███▄█▄█▄███░█▒████▓▓░
   ⠀ ⠀⠀⠀⠀⠐⠡⡑⠜⢎⢗⢕⢘⢜⢜⢜⠜⠕⠡⠡⡈⠀⠀⠀⠀⠀⠀ ⠀⠀⠀   ░█░███▀█▀█▀█░█▀▀▒█████▓░
   ⠀ ⠀⠀⠀⠀⠀⠁⡢⢀⠈⠨⣂⡐⢅⢕⢐⠁⠡⠡⢁⠀⠀⠀⠀⠀⠀⠀  ⠀⠀    ░█░▀▄█▄█▄█▄▀▒▒▒▒█████▓░
 ⠀ ⠀ ⠀ ⠀⠀⠀⠀⢈⠢⠀⡀⡐⡍⢪⢘⠀⠀⠡⡑⡀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀     ░████░░░░░░▒▓▓███████▓░
 ⠀⠀  ⠀ ⠀⠀⠀⠀⠀⠨⢂⠀⠌⠘⢜⠘⠀⢌⠰⡈⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀     ░▓███▒▄▄▄▄▒▒▒▒████████░
 ⠀⠀⠀   ⠀⠀⠀⠀⠀⠀⢑⢸⢌⢖⢠⢀⠪⡂                      ░▓▓██▒▓███████████████░

                  EVAN                                 MADISON

 ############################################################################'''

if (False):
    # Update PATH_NAME so it points to the local bucket on your computer
    #PATH_NAME = "C:\\Users\\madle\\Documents\\EE148"
    PATH_NAME = "..\\..\\data\\midair"
    
    ''' TODO: Split up entire dataset into chunks of 50, then randomly assign those
        chunks to testing, training, and validation sets'''
    
    ''' TODO: Split up the dataset into training, testing, and validation datasets'''
    
    # dataset = midairDataset(main_path = PATH_NAME,
    #                 # change PATH_NAME here to the appropriate data directory
    #                         data_path = os.path.join(PATH_NAME,''),
    #                         transform = transforms.Compose([
    #                                     transforms.ToTensor()
    #                                     ]))
    dataset = midairDataset(main_path = PATH_NAME,
                    # change PATH_NAME here to the appropriate data directory
                            data_path = os.path.join(PATH_NAME,''))
    objects = 0
    signs = 0
    posts = 0
    pictures_with = 0
    for i in range(len(dataset)):
        image, data = dataset[i]
        if len(data) != 0:
            pictures_with += 1
            for j in range(len(data)):
                objects += 1
                if data[j][0] == 0:
                    posts += 1
                else:
                    signs += 1
    print('Number of images with objects: ', pictures_with)
    print('Fraction of images with objects: ', pictures_with/len(dataset))
    print('Number of objects: ', objects)
    print('Number of signs: ', signs)
    print('Number of posts: ', posts)   
