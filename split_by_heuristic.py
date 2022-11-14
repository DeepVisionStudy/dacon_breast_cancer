import cv2
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from collections import Counter

from utils import set_seeds

PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')

def main(file_name):
    df = pd.read_csv(osp.join(PATH_DATA, file_name))

    window_size = 10  # kernel_size
    step_size = 5  # stride
    percent_thres = 85  # 밝은 영역 기준
    smooth_value = 5
    top_k = 5  # top_k 만큼 평균 구하기
    max_k = 4  # max_k = 최대 split_value

    split_column = []
    for idx in tqdm(range(len(df))):
        img_path = df['img_path'][idx]
        img_path = osp.join(PATH_DATA, img_path).replace('\\.','')

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
    
    df['split'] = split_column
    new_file_name = file_name.split('.')[0] + '_heuristic.csv'
    df.to_csv(osp.join(PATH_DATA, new_file_name), index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    for csv_file in ['train.csv', 'test.csv']:
        set_seeds(42)
        main(csv_file)