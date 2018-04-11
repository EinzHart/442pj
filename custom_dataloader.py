import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class MyCustomDataset(Dataset):
    def __init__(self, dir, training_flag = True, transforms=None):
        self.root_dir = dir
        self.filenames = ['{}.png'.format(i) for i in range(len(os.listdir(dir+'color/')))]
        self.transforms = transforms
        self.training_flag = training_flag


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        color_img = cv2.imread(os.path.join(self.root_dir,'color/',self.filenames[index]),
                                cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(os.path.join(self.root_dir,'mask/',self.filenames[index]),
                                cv2.IMREAD_GRAYSCALE)
        if self.training_flag:
            normal_img = cv2.imread(os.path.join(self.root_dir,'normal/',self.filenames[index]))

        # keep it 3-D
        color_img.shape = color_img.shape + (1,)
        mask_img.shape = mask_img.shape + (1,)

        if self.transforms is not None:
            color_img = self.transforms(color_img)
            mask_img = self.transforms(mask_img)
            if self.training_flag:
                normal_img = self.transforms(normal_img)
                normal_img = (normal_img - 0.5) * 2
                # normals now FloatTensor in range -1 to 1
                # with size C x H x W
        mask_img = (mask_img != 0).type(torch.FloatTensor)

        if self.training_flag:
            return {'color':color_img, 'mask':mask_img, 'normal':normal_img}
        else:
            return {'color':color_img, 'mask':mask_img}
# class MyCustomDataset()
