import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

import torch

from dataset import create_data_loader, preprocess_df
from model import ClassificationModel
from utils import set_seeds, load_config, str2bool


def inference(model, test_loader, device, cls_last_sigmoid):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for img, tab in tqdm(iter(test_loader)):
            img = img.float().to(device)
            tab = tab.float().to(device)
            
            model_pred = model(img, tab)
            if not cls_last_sigmoid:
                model_pred = torch.sigmoid(model_pred)
            model_pred = model_pred.squeeze(1).to('cpu')
            
            preds += model_pred.tolist()

    return preds


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--df', type=str, default='test')
    parser.add_argument('--mode', type=str, default='test') # test hard valid
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument('--vflip', action='store_true')
    parser.add_argument('--ckpt_list', nargs='+') # exp0/best.pt
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--resize_by_split', action='store_true')
    parser.add_argument('--transform_type', type=str, default='resize')
    parser.add_argument('--pred_thres', type=float, default=0.5)
    parser.add_argument('--remark', type=str, default='')
    args = parser.parse_args()
    
    if args.mode == 'valid':
        assert args.df in ['train', 'train_5fold', 'train_heuristic_5fold']
    else:
        assert args.df in ['test', 'test_heuristic']

    args.base_dir = './'
    args.data_dir = osp.join(args.base_dir, 'data')
    args.submit_dir = osp.join(args.base_dir, 'submit')
    os.makedirs(args.submit_dir, exist_ok=True)
    
    return args


def main(args, train_args, hflip=False, vflip=False):
    args.device = torch.device("cuda:0")

    df = pd.read_csv(osp.join(args.data_dir, f'{args.df}.csv'))
    if args.mode == 'valid':
        df = preprocess_df(df, train_args.df_ver, drop_row=True)
        df = df[df["kfold"] == train_args.fold].reset_index(drop=True)
    else:
        df = preprocess_df(df, train_args.df_ver, drop_row=False)
    
    test_loader = create_data_loader(
        df, 'infer', args.img_size, data_dir=args.data_dir,
        hflip=hflip, vflip=vflip, norm_type=train_args.norm_type,
        transform_type=args.transform_type, resize_by_split=args.resize_by_split,
    )
    
    # exp235
    # args.tab_model = 'drop40'
    model = ClassificationModel(train_args).to(args.device)
    model.load_state_dict(torch.load(osp.join(args.work_dir, args.ckpt)))

    preds = inference(model, test_loader, args.device, train_args.cls_last_sigmoid)
    if args.mode in ['valid', 'hard']:
        preds = np.where(np.array(preds) > args.pred_thres, 1, 0)
    if args.mode == 'valid':
        labels = df[df['kfold'] == train_args.fold]['N_category'].tolist()
        valid_f1 = metrics.f1_score(y_true=labels, y_pred=preds, average='macro', zero_division=1)
        print(f'hflip={bool(hflip)} vflip={bool(vflip)} valid_f1:{valid_f1:.4f}')
        
    return preds


if __name__ == '__main__':
    args = get_parser()
    
    submit_file_name = args.mode
    if args.hflip:
        submit_file_name += '_hflip'
    if args.vflip:
        submit_file_name += '_vflip'
    if args.resize_by_split:
        submit_file_name += '_resizeBySplit'
    if args.transform_type != 'resize':
        submit_file_name += f'_{args.transform_type}'
    if args.img_size != 512:
        submit_file_name += f'_img{args.img_size}'
    if args.remark != '':
        submit_file_name += f'_{args.remark}'

    # preds
    ensemble = []
    ensemble_cnt = 0
    for idx in range(len(args.ckpt_list)):
        args.exp, args.ckpt = args.ckpt_list[idx].split('/')
        args.work_dir = osp.join('work_dirs', args.exp) # work_dirs/exp0
        
        submit_file_name += f'_{args.exp}'
        if args.ckpt.split('.')[0] != 'best':
            submit_file_name += args.ckpt.split('.')[0]
        
        set_seeds(args.seed)
        train_args = load_config(osp.join(args.work_dir, 'config.yaml'))
        
        for hflip in range(args.hflip + 1):
            for vflip in range(args.vflip + 1):
                preds = main(args, train_args, hflip, vflip)
                ensemble.append(preds)
                ensemble_cnt += 1
                
    if args.mode in ['test', 'hard']:
        # ensemble
        if args.mode == 'hard':
            ensemble = np.array(ensemble)
            preds = [np.argmax(np.bincount(ensemble[:,i])) for i in range(ensemble.shape[1])]
        elif args.mode == 'test':
            if ensemble_cnt > 1:
                ensemble = np.sum(ensemble, axis=0)
            elif ensemble_cnt == 1:
                ensemble = ensemble[0]
            preds = np.where(ensemble > args.pred_thres * ensemble_cnt, 1, 0)
        
        # output
        submit = pd.read_csv(osp.join(args.data_dir, 'sample_submission.csv'))
        submit['N_category'] = preds
        submit_path = osp.join(args.submit_dir, f'{submit_file_name}.csv')
        submit.to_csv(submit_path, index=False)