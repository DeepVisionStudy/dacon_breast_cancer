import wandb
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from dataset import create_data_loader
from model import ClassificationModel
from utils import set_seeds, get_exp_dir, save_config
from set_wandb import wandb_init


def run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, args):
    scaler = GradScaler()
    best_score = 0.0
    
    for epoch in range(1, args.epochs+1):
        print('-' * 10)
        print(f'Epoch {epoch}/{args.epochs}')

        model.train()
        train_loss = []
        for img, tab, label in tqdm(iter(train_loader)):
            img = img.float().to(args.device)
            tab = tab.float().to(args.device)
            label = label.float().to(args.device)
            
            optimizer.zero_grad()
            
            if args.amp:
                with autocast():
                    model_pred = model(img, tab)
                    loss = criterion(model_pred, label.reshape(-1,1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            else:
                model_pred = model(img, tab)
                loss = criterion(model_pred, label.reshape(-1,1))
                loss.backward()
                optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss, val_score = validation(model, criterion, val_loader, args)
        print(f'Epoch [{epoch}] Train Loss : [{np.mean(train_loss):.4f}] Val Loss : [{val_loss:.4f}] Val Score : [{val_score:.4f}]')
        
        if args.sched == 'reduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if best_score < val_score:
            best_score = val_score
            torch.save(model.state_dict(), osp.join(args.work_dir, 'best.pt'))
        
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), osp.join(args.work_dir, f'epoch{epoch}.pt'))
        
        wandb.log({
            'train/loss': round(np.mean(train_loss),4), 'valid/loss': round(val_loss,4), 'valid/f1': round(val_score,4),
        })


def validation(model, criterion, val_loader, args):
    model.eval()
    pred_labels = []
    true_labels = []
    val_loss = []

    with torch.no_grad():
        for img, tabular, label in tqdm(iter(val_loader)):
            true_labels += label.tolist()
            
            img = img.float().to(args.device)
            tabular = tabular.float().to(args.device)
            label = label.float().to(args.device)
            
            model_pred = model(img, tabular)
            
            loss = criterion(model_pred, label.reshape(-1,1))
            
            val_loss.append(loss.item())
            
            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
    
    pred_labels = np.where(np.array(pred_labels) > args.pred_thres, 1, 0)
    val_score = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    return np.mean(val_loss), val_score


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--work_dir', type=str, default='./work_dirs')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--pred_thres', type=float, default=0.5)
    
    # Data
    parser.add_argument('--df', type=str, default='train_heuristic_5fold')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Model
    parser.add_argument('--img_model', type=str, default="efficientnet_b0")
    parser.add_argument('--tab_model', type=str, default="baseline")
    parser.add_argument('--tab_init_feat', type=int, default=24)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--crit', type=str, default="bce")
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--sched', type=str, default="reduce")
    
    args = parser.parse_args()
    args.base_dir = './'
    args.data_dir = osp.join(args.base_dir, 'data')
    args.work_dir = get_exp_dir(args.work_dir)
    args.config_dir = osp.join(args.work_dir, 'config.yaml')
    return args


def main(args):
    args.device = torch.device("cuda:0")
    
    df = pd.read_csv(osp.join(args.data_dir, f'{args.df}.csv'))
    df = df.fillna(-1)

    train_df = df[df["kfold"] != args.fold].drop(columns=['kfold']).reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].drop(columns=['kfold']).reset_index(drop=True)

    train_loader = create_data_loader(train_df, args.img_size, args.batch_size, args.num_workers, 'train', args.data_dir)
    valid_loader = create_data_loader(valid_df, args.img_size, args.batch_size, args.num_workers, 'valid', args.data_dir)

    model = ClassificationModel(args).to(args.device)
    
    if args.crit == 'bce':
        criterion = nn.BCEWithLogitsLoss().to(args.device)
    
    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.sched == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=1, threshold_mode='abs', min_lr=1e-8, verbose=True)
    elif args.sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=1, eta_min=0, verbose=True)
    
    run_train(model, train_loader, valid_loader, criterion, optimizer, scheduler, args)


if __name__ == '__main__':
    for fold in [1,2,3,4,5]:
        args = get_parser()
        args.fold = fold
        save_config(args, args.config_dir)
        set_seeds(args.seed)
        wandb_init(args)
        main(args)