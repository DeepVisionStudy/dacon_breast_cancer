import pandas as pd
import os.path as osp
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import set_seeds

PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')

def main(file_name, ver):
    n_splits = 5
    if ver == 1:
        folds = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    elif ver == 2:
        folds = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    df = pd.read_csv(osp.join(PATH_DATA, file_name))
    X = df.values
    if ver == 1:
        y = df['N_category']
    elif ver == 2:
        y = df[['N_category','진단명','DCIS_or_LCIS_여부']]
    split_idx = list(folds.split(X, y))

    df['kfold'] = -1
    for i in range(n_splits):
        _, valid_idx = split_idx[i]
        valid = df.iloc[valid_idx]
        condition = df.ID.isin(valid.ID) == True
        df.loc[df[condition].index.to_list(), 'kfold'] = i+1  # 5fold : 1~5
    if ver == 1:
        new_file_name = file_name.split('.')[0] + f'_{n_splits}fold.csv'
    elif ver == 2:
        new_file_name = file_name.split('.')[0] + f'_{n_splits}foldver2.csv'
    df.to_csv(osp.join(PATH_DATA, new_file_name), index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    set_seeds(42)
    file_name = 'train_heuristic.csv'
    main(file_name, 2)