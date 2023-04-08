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
    def __init__(self, medical_df, labels, mode, transforms=None, data_dir='./data',
                 transform_type='resize', resize_by_split=False, img_size=512):
        self.medical_df = medical_df
        self.labels = labels
        self.mode = mode
        self.transforms = transforms
        self.data_dir= data_dir
        self.transform_type = transform_type
        self.resize_by_split = resize_by_split
        self.img_size = img_size

        self.drop_col = ['ID', 'img_path', '수술연월일']
        for col in ['N_category', 'split', 'mask_path', 'kfold']:
            if col in self.medical_df.columns:
                self.drop_col.append(col)
        
    def __getitem__(self, index):
        img_path = self.medical_df['img_path'].iloc[index]
        img_path = osp.join(self.data_dir, img_path).replace('\\.','')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # split 중 random하게 하나만 선택
        if self.mode == 'train':
            split = self.medical_df['split'].iloc[index]
            new_w = img.shape[1] // split
            randn = np.random.randint(split)
            img = np.array(img)[:, new_w*randn:new_w*(randn+1), :]
        elif self.mode == 'infer':
            if self.resize_by_split:
                split = self.medical_df['split'].iloc[index]
                transform = []
                if self.transform_type == 'resize':
                    transform.append(A.Resize(self.img_size, self.img_size*split))
                transform = A.Compose(transform)
                img = transform(image=img)['image']

        # image augmentation
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        # output
        tab = self.medical_df.drop(columns=self.drop_col).iloc[index]
        tab = torch.Tensor(tab)
        # # tab confidence
        # tab_conf = list(tab).count(-1) / len(list(tab))
        # tab = (torch.Tensor(tab), torch.as_tensor(tab_conf))
        if self.labels is not None:
            # # tab augmentation
            # if self.mode == 'train':
            #     tab[0] += np.random.randint(5) - 2  # 나이
            #     if tab[4] != -1:  # 암의 장경
            #         randn = np.random.randint(5)
            #         if tab[4] + randn - 2 > 0:
            #             tab[4] += randn - 2
            #     if tab[16] != -1:  # KI-67_LI_percent
            #         randn = np.random.randint(3)
            #         if tab[16] + randn - 1 > 0:
            #             tab[16] += randn - 1
            return img, tab, self.labels[index]
        else:
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


def create_data_loader(df, mode, img_size, batch_size=1, num_workers=0,
                       data_dir='./data', hflip=False, vflip=False, norm_type='baseline',
                       transform_type='resize', resize_by_split=False):
    
    if norm_type == 'baseline':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif norm_type == 'custom':
        mean, std = (0.9306, 0.9071, 0.9253), (0.0524, 0.1001, 0.0612)

    if mode == 'train':
        resize_size = int(img_size * 1.2)
        transforms = A.Compose([
            A.Resize(resize_size, resize_size),
            A.RandomResizedCrop(img_size, img_size, scale=(0.5,1.0), ratio=(1.0, 1.0)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            # A.RandomBrightnessContrast(),
            # A.CLAHE(),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            # A.CoarseDropout(),
            ToTensorV2()
        ])
        dataset = CustomDataset(df, df['N_category'], mode, transforms, data_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    elif mode == 'valid':
        transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])
        dataset = CustomDataset(df, df['N_category'], mode, transforms, data_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif mode == 'infer':
        transforms = []
        if not resize_by_split:
            if transform_type == 'resize':
                transforms.append(A.Resize(img_size, img_size))
        if hflip:
            transforms.append(A.HorizontalFlip(p=1.0))
        if vflip:
            transforms.append(A.VerticalFlip(p=1.0))
        transforms.extend([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])
        transforms = A.Compose(transforms)

        dataset = CustomDataset(df, None, mode, transforms, data_dir, transform_type, resize_by_split, img_size)
        # dataset.get_split_value()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader


def preprocess_df(df, ver=1, drop_row=False):
    if drop_row:
        drop_idx = []
        drop_idx.extend(df[df['PR_Allred_score'] > 8].index.to_list())
        drop_idx.extend(df[pd.isna(df['ER'])].index.to_list())
        drop_idx.extend(df[pd.isna(df['T_category'])].index.to_list())
        drop_idx.extend(df[pd.isna(df['HER2'])].index.to_list())
        df = df.drop(drop_idx).reset_index(drop=True)
    
    drop_col = ['DCIS_or_LCIS_type', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation']
    df = df.drop(columns=drop_col).reset_index(drop=True)

    df['KI-67_LI_percent'] = df['KI-67_LI_percent'].apply(lambda x:x if np.isnan(x) else int(x))
    
    if ver == 7:
        df['HG'] = df['HG'].replace(4,0)
        df['HG_score_1'] = df['HG_score_1'].replace(4,0)
        df['HG_score_2'] = df['HG_score_2'].replace(4,0)
        df['HG_score_3'] = df['HG_score_3'].replace(4,0)

    df = df.fillna(-1)
    # df.to_csv(osp.join('./data', 'after_preprocess.csv'), index=True, encoding="utf-8-sig")

    return df
