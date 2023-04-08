import numpy as np
import pandas as pd
import os.path as osp
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import set_seeds

PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')


def main(file_name):
    df = pd.read_csv(osp.join(PATH_DATA, file_name))
    
    if 'train' in file_name:
        drop_idx = []
        drop_idx.extend(df[df['PR_Allred_score'] > 8].index.to_list())
        drop_idx.extend(df[pd.isna(df['ER'])].index.to_list())
        drop_idx.extend(df[pd.isna(df['T_category'])].index.to_list())
        drop_idx.extend(df[pd.isna(df['HER2'])].index.to_list())
        df = df.drop(drop_idx)
    
    drop_col = ['DCIS_or_LCIS_type', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation']
    df = df.drop(columns=drop_col).reset_index(drop=True)
    
    df['KI-67_LI_percent'] = df['KI-67_LI_percent'].apply(lambda x:x if np.isnan(x) else int(x))
    
    df = df.fillna(-1)

    std_scaler = OneHotEncoder()
    
    df = pd.concat([df[['ID','img_path','mask_path','나이','수술연월일']]])

    print(df.info())


if __name__ == '__main__':
    set_seeds(42)
    file_name = 'train_heuristic_5fold.csv'
    main(file_name)