from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image

import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from defs import class_to_num



class LeavesData(Dataset):
    def __init__(self, csv_name, file_path, mode='train', valid_ratio=0.2):

        self.file_path = file_path
        self.mode = mode

        self.data = pd.read_csv(file_path + csv_name)  #header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data['image'])
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # if self.mode == 'train':
        #     transform = albumentations.Compose([
        #         albumentations.Resize(320, 320),
        #         albumentations.HorizontalFlip(p=0.5),
        #         albumentations.VerticalFlip(p=0.5),
        #         albumentations.Rotate(limit=180, p=0.7),
        #         albumentations.RandomBrightnessContrast(),
        #         albumentations.ShiftScaleRotate(
        #             shift_limit=0.25, scale_limit=0.1, rotate_limit=0
        #         ),
        #         albumentations.Normalize(
        #             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        #             max_pixel_value=255.0, always_apply=True
        #         ),
        #         ToTensorV2(p=1.0),
        #     ])
        # else:
        #     transform = albumentations.Compose([
        #         albumentations.Resize(320, 320),
        #         albumentations.Normalize(
        #             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        #             max_pixel_value=255.0, always_apply=True
        #         ),
        #         ToTensorV2(p=1.0),

        #     ])
        
        self.transform = transform
        
        if mode == 'train':
            # 第一列包含图像文件的名称
            self.images = np.asarray(self.data.iloc[0:self.train_len, 0])  #self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.labels = np.asarray(self.data.iloc[0:self.train_len, 1])
        elif mode == 'valid':
            self.images = np.asarray(self.data.iloc[self.train_len:, 0])  
            self.labels = np.asarray(self.data.iloc[self.train_len:, 1])
        elif mode == 'test':
            self.images = np.asarray(self.data.iloc[0:, 0])
        
        self.length = len(self.images)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.length))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        image_path = self.images[index]

        # 读取图像文件
        image = Image.open(self.file_path + image_path)
        image = self.transform(image)

        # image = cv2.imread(self.file_path + image_path)
        # image = self.transform(image = image)['image']

        
        if self.mode == 'test':
            return image
        else:
            # 得到图像的 string label
            label = self.labels[index]
            # number label
            number_label = class_to_num[label]

            return image, number_label  #返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.length