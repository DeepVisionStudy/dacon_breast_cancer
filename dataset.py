import cv2
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CustomDataset(Dataset):
    def __init__(self, medical_df, labels, mode, transforms=None, data_dir='./data'):
        self.medical_df = medical_df
        self.labels = labels
        self.mode = mode
        self.transforms = transforms
        self.data_dir= data_dir
        
    def __getitem__(self, index):
        img_path = self.medical_df['img_path'].iloc[index]
        img_path = osp.join(self.data_dir, img_path).replace('\\.','')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # if self.mode == 'train':
        #     split = self.medical_df['split'].iloc[index]
        #     new_w = img.shape[1] // split
        #     randn = np.random.randint(split)
        #     img = np.array(img)[:, new_w*randn:new_w*(randn+1), :]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
                
        if self.labels is not None:  # train / valid
            tab = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', 'mask_path', '수술연월일', 'split']).iloc[index])
            label = self.labels[index]
            return img, tab, label
        else:  # test
            tab = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', '수술연월일', 'split']).iloc[index])
            return img, tab
        
    def __len__(self):
        return len(self.medical_df)

    def get_split_value(self):
        split_column = []

        window_size = 10  # kernel_size
        step_size = 5  # stride
        percent_thres = 85  # 밝은 영역 기준
        smooth_value = 5
        top_k = 5  # top_k 만큼 평균 구하기
        max_k = 4  # max_k = 최대 split_value

        for idx in tqdm(range(len(self.medical_df))):
            img_path = self.medical_df['img_path'][idx]
            img_path = osp.join(self.data_dir, img_path).replace('\\.','')

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            arr = img.mean(axis=(0,2))
            arr = pd.Series(arr).rolling(window=window_size, step=step_size).mean().iloc[window_size-1:].values

            is_over = np.percentile(arr, percent_thres, method='closest_observation')
            arr = np.cumsum(arr > is_over)
            arr = (arr/smooth_value).astype(np.uint8)

            counter = Counter(arr).most_common(top_k)
            count_mean = np.mean([count for value, count in counter])
            split_value = len([True for value, count in counter[:max_k] if count > count_mean])

            split_column.append(split_value)
        
        self.medical_df['split'] = split_column


def create_data_loader(df, img_size, batch_size, num_workers, mode, data_dir, hflip=False):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == 'train':
        labels = df['N_category']
        df = df.drop(columns=['N_category'])
        transforms = A.Compose([
            A.LongestMaxSize(max_size=img_size*2),
            A.PadIfNeeded(min_height=img_size, min_width=img_size),
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])
        dataset = CustomDataset(df, labels, mode, transforms, data_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    elif mode == 'valid':
        labels = df['N_category']
        df = df.drop(columns=['N_category'])
        transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])
        dataset = CustomDataset(df, labels, mode, transforms, data_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif mode == 'test':
        if hflip:
            transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2()
            ])
        else:
            transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2()
            ])
        dataset = CustomDataset(df, None, mode, transforms, data_dir)
        # dataset.get_split_value()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    return loader