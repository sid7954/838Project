import csv
import cv2
import os
import numpy as np
from tqdm import tqdm_notebook as tqdm

import urllib
import cv2, csv
from matplotlib import pyplot as plt
 
    
import torch, torchvision
import torch.utils.data
import pandas as pd
from torch.utils.data.dataset import Dataset  
from torchvision import transforms, utils
from PIL import Image

    
class FacesDataset(Dataset):
    def __init__(self, root_dir, csv_file, transforms=None):
        self.labels_frame = pd.read_csv(os.path.join(csv_file))
        self.root_dir = root_dir
        # self.transform = transforms.Compose([transforms.ColorJitter(),
        #                                      transforms.RandomAffine(degrees, translate=None, scale=None, shear=None,),
        #                                      ToTensor()])
        self.transform=transforms
        self.num_celebs = int(self.labels_frame.iloc[len(self.labels_frame)-1, 2])+1
        self.labels_frame.drop(0)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name)
        
        def one_hot(idx):
            a = np.zeros(self.num_celebs, dtype=int)
            a[idx] = 1
            return int(idx) #torch.from_numpy(np.array([idx]))
        
        #print(type(image))
        if self.transform is not None:
            image=self.transform(image)

        sample =  (image, one_hot(self.labels_frame.iloc[idx, 2]))

        #if self.transform is not None:
        #    sample = self.transform(sample)

        return sample
    

    def __len__(self):
        return len(self.labels_frame)


    
    
    
