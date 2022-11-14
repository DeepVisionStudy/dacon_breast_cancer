import pandas as pd
import os.path as osp
from sklearn.model_selection import StratifiedKFold

from utils import set_seeds

PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')

def main(file_name):
    n_splits = 5
    folds = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    
    df = pd.read_csv(osp.join(PATH_DATA, file_name))
    split_idx = list(folds.split(df.values, df['N_category']))

    df['kfold'] = -1
    for i in range(n_splits):
        _, valid_idx = split_idx[i]
        valid = df.iloc[valid_idx]
        condition = df.ID.isin(valid.ID) == True
        df.loc[df[condition].index.to_list(), 'kfold'] = i+1  # 5fold : 1~5

    new_file_name = file_name.split('.')[0] + f'_{n_splits}fold.csv'
    df.to_csv(osp.join(PATH_DATA, new_file_name), index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    set_seeds(42)
    file_name = 'train_heuristic.csv'
    main(file_name)