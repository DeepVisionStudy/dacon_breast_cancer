import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from dataset import create_data_loader
from model import ClassificationModel
from utils import set_seeds, load_config


def inference(model, test_loader, device):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for img, tab in tqdm(iter(test_loader)):
            img = img.float().to(device)
            tab = tab.float().to(device)
            
            model_pred = model(img, tab)
            model_pred = model_pred.squeeze(1).to('cpu')
            
            preds += model_pred.tolist()

    return preds


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--df', type=str, default='test')
    parser.add_argument('--mode', type=str, default='test') # test soft hard
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument('--vflip', action='store_true')
    parser.add_argument('--ckpt_list', nargs='+') # exp0/best.pt
    args = parser.parse_args()
    args.base_dir = './'
    args.data_dir = osp.join(args.base_dir, 'data')
    args.submit_dir = osp.join(args.base_dir, 'submit')
    os.makedirs(args.submit_dir, exist_ok=True)
    return args


def main(args, train_args, hflip=False, vflip=False):
    args.device = torch.device("cuda:0")
    
    df = pd.read_csv(osp.join(args.data_dir, f'{args.df}.csv'))
    # 결측값 처리
    df = df.fillna(-1)

    test_loader = create_data_loader(
        df, 'test', train_args.img_size, data_dir=args.data_dir, hflip=hflip, vflip=vflip)
    
    model = ClassificationModel(train_args).to(args.device)
    model.load_state_dict(torch.load(osp.join(args.work_dir, args.ckpt)))

    preds = inference(model, test_loader, args.device)
    # hard ensemble은 label 필요
    if args.mode == 'hard':
        preds = np.where(np.array(preds) > train_args.pred_thres, 1, 0)
    return preds


if __name__ == '__main__':
    args = get_parser()
    
    submit_file_name = args.mode
    if args.hflip:
        submit_file_name += '_hflip'
    if args.vflip:
        submit_file_name += '_vflip'

    ensemble = []
    
    for idx in range(len(args.ckpt_list)):
        args.exp, args.ckpt = args.ckpt_list[idx].split('/')
        args.work_dir = osp.join('work_dirs', args.exp) # work_dirs/exp0
        
        submit_file_name += f'_{args.exp}'
        # defalut : best.pt
        if args.ckpt.split('.')[0] != 'best':
            submit_file_name += args.ckpt.split('.')[0]
        
        set_seeds(args.seed)
        train_args = load_config(osp.join(args.work_dir, 'config.yaml'))
        
        for hflip in range(args.hflip + 1):
            for vflip in range(args.vflip + 1):
                preds = main(args, train_args, hflip, vflip)
                ensemble.append(preds)

    # probability를 sum해서 모았기 때문에, sum한 만큼 pred_thres를 맞춰줌
    pred_thres = train_args.pred_thres * len(args.ckpt_list)
    if args.hflip:
        pred_thres *= 2.0
    if args.vflip:
        pred_thres *= 2.0

    # 모은 ensemble을 preds (label 형태)로 만들기
    if args.mode == 'test':
        if args.hflip:
            ensemble = np.sum(ensemble, axis=0)
        else:
            ensemble = ensemble[0]
        preds = np.where(ensemble > pred_thres, 1, 0)
    
    elif args.mode == 'soft':
        ensemble = np.sum(ensemble, axis=0)
        preds = np.where(ensemble > pred_thres, 1, 0)

    elif args.mode == 'hard':
        ensemble = np.array(ensemble)
        preds = [np.argmax(np.bincount(ensemble[:,i])) for i in range(ensemble.shape[1])]
    
    # 제출 파일
    submit = pd.read_csv(osp.join(args.data_dir, 'sample_submission.csv'))
    submit['N_category'] = preds
    submit_path = osp.join(args.submit_dir, f'{submit_file_name}.csv')
    submit.to_csv(submit_path, index=False)