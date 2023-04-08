import warnings
warnings.filterwarnings(action='ignore')

import wandb
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from sklearn import metrics
from adamp import AdamP
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from dataset import create_data_loader, preprocess_df
from model import ClassificationModel
from utils import set_seeds, get_exp_dir, save_config, str2bool
from set_wandb import wandb_init


def run_train(model, train_loader, val_loader, criterion, crit_aux, optimizer, scheduler, args):
    if val_loader == None:
        only_train = True
        assert args.sched != 'reduce'
    else:
        only_train = False
    
    scaler = GradScaler()
    best_score = 0.0
    best_loss = 999999.0
    
    for epoch in range(1, args.epochs+1):
        print('-' * 10)
        print(f'Epoch {epoch}/{args.epochs}')
        
        if epoch == 1:
            model.img_feature_extractor.requires_grad('back', False)
        if epoch == args.img_back_freeze + 1:
            model.img_feature_extractor.requires_grad('back', True)
        
        model.train()

        pred_labels_proba = []
        true_labels = []
        train_loss = []
        for img, tab, label in tqdm(iter(train_loader)):
            true_labels += label.tolist()

            img = img.float().to(args.device)
            tab = tab.float().to(args.device)
            label = label.float().to(args.device)
            
            optimizer.zero_grad()
            
            if args.amp:
                with autocast():
                    if args.cls_return == 'baseline':
                        model_pred = model(img, tab)
                    elif args.cls_return == 'aux':
                        model_pred, img_pred, tab_pred = model(img, tab)
                    
                    loss = 0
                    for i in range(len(criterion)):
                        loss += args.crit_coef[i] * criterion[i](model_pred, label.reshape(-1,1))
                    if len(crit_aux):
                        loss += args.crit_aux_coef[0] * crit_aux[0](img_pred, label.reshape(-1,1))
                        loss += args.crit_aux_coef[1] * crit_aux[1](tab_pred, label.reshape(-1,1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            train_loss.append(loss.item())

            if not args.cls_last_sigmoid:
                model_pred = torch.sigmoid(model_pred)
            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels_proba += model_pred.tolist()
        
        pred_labels = np.where(np.array(pred_labels_proba) > args.pred_thres, 1, 0)
        
        train_acc = metrics.accuracy_score(y_true=true_labels, y_pred=pred_labels)
        train_f1 = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro', zero_division=1)
        train_pr = metrics.precision_score(y_true=true_labels, y_pred=pred_labels, average='macro', zero_division=1)
        train_rc = metrics.recall_score(y_true=true_labels, y_pred=pred_labels, average='macro', zero_division=1)
        train_roc_auc = metrics.roc_auc_score(y_true=true_labels, y_score=pred_labels_proba, average='macro')

        if not only_train:
            val_loss, val_acc, val_f1, val_pr, val_rc, val_roc_auc = validation(model, criterion, crit_aux, val_loader, args)
        
        wandb.log({'learning_rate': scheduler.optimizer.param_groups[0]['lr']}, commit=False)
        if args.sched == 'reduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if not only_train:
            if args.save_metric == 'f1':
                if best_score < val_f1:
                    best_score = val_f1
                    torch.save(model.state_dict(), osp.join(args.work_dir, 'best.pt'))
            elif args.save_metric == 'loss':
                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), osp.join(args.work_dir, 'best.pt'))
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), osp.join(args.work_dir, f'epoch{epoch}.pt'))

        print(f'Train Loss : [{np.mean(train_loss):.4f}] Train Acc : [{train_acc:.4f}] Train f1 : [{train_f1:.4f}] Train PR : [{train_pr:.4f}] Train RC : [{train_rc:.4f}] Train ROC-AUC : [{train_roc_auc:.4f}]')
        if not only_train:
            print(f'Val Loss : [{val_loss:.4f}] Val Acc : [{val_acc:.4f}] Val f1 : [{val_f1:.4f}] Val PR : [{val_pr:.4f}] Val RC : [{val_rc:.4f}] Val ROC-AUC : [{val_roc_auc:.4f}]')
            wandb.log({
                'valid/loss': round(val_loss,4), 'valid/acc': round(val_acc,4),
                'valid/f1': round(val_f1,4), 'valid/pr': round(val_pr,4), 'valid/rc': round(val_rc,4), 'valid/roc_auc_score': round(val_roc_auc,4),
            }, commit=False)
        wandb.log({'train/loss': round(np.mean(train_loss),4), 'train/acc': round(train_acc,4), 'train/f1': round(train_f1,4), 'train/pr': round(train_pr,4), 'train/rc': round(train_rc,4), 'train/roc_auc_score': round(train_roc_auc,4)})


def validation(model, criterion, crit_aux, val_loader, args):
    model.eval()
    pred_labels_proba = []
    true_labels = []
    val_loss = []

    with torch.no_grad():
        for img, tab, label in tqdm(iter(val_loader)):
            true_labels += label.tolist()
            
            img = img.float().to(args.device)
            tab = tab.float().to(args.device)
            label = label.float().to(args.device)
            
            if args.cls_return == 'baseline':
                model_pred = model(img, tab)
            elif args.cls_return == 'aux':
                model_pred, img_pred, tab_pred = model(img, tab)
            
            loss = 0
            for i in range(len(criterion)):
                loss += args.crit_coef[i] * criterion[i](model_pred, label.reshape(-1,1))
            if len(crit_aux):
                loss += args.crit_aux_coef[0] * crit_aux[0](img_pred, label.reshape(-1,1))
                loss += args.crit_aux_coef[1] * crit_aux[1](tab_pred, label.reshape(-1,1))
            
            val_loss.append(loss.item())
            
            if not args.cls_last_sigmoid:
                model_pred = torch.sigmoid(model_pred)
            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels_proba += model_pred.tolist()
    
    pred_labels = np.where(np.array(pred_labels_proba) > args.pred_thres, 1, 0)
    val_acc = metrics.accuracy_score(y_true=true_labels, y_pred=pred_labels)
    val_f1 = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro', zero_division=1)
    val_pr = metrics.precision_score(y_true=true_labels, y_pred=pred_labels, average='macro', zero_division=1)
    val_rc = metrics.recall_score(y_true=true_labels, y_pred=pred_labels, average='macro', zero_division=1)
    val_roc_auc = metrics.roc_auc_score(y_true=true_labels, y_score=pred_labels_proba, average='macro')

    return np.mean(val_loss), val_acc, val_f1, val_pr, val_rc, val_roc_auc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_metric', type=str, default='f1')
    parser.add_argument('--work_dir', type=str, default='./work_dirs')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--fold', type=int, nargs='+', default=[1,2,3,4,5])  # 0 : all_train
    parser.add_argument('--amp', type=str2bool, default=True)
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--pred_thres', type=float, default=0.5)
    
    # Data
    parser.add_argument('--df', type=str, default='train_heuristic_5fold')  # train_5fold / train_heuristic_5fold
    parser.add_argument('--df_ver', type=int, default=7)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--norm_type', type=str, default='baseline')  # baseline / custom
    
    # Model
    parser.add_argument('--img_model', type=str, default="efficientnet_b0")
    parser.add_argument('--img_back_freeze', type=int, default=0)
    parser.add_argument('--img_last_feat', type=int, default=1)
    parser.add_argument('--tab_model', type=str, default="drop20")
    parser.add_argument('--tab_init_feat', type=int, default=19)
    parser.add_argument('--tab_last_feat', type=int, default=1)
    parser.add_argument('--cls_model', type=str, default="baseline")
    parser.add_argument('--cls_fusion', type=str, default="cat")
    parser.add_argument('--cls_return', type=str, default="baseline")  # baseline / aux
    parser.add_argument('--cls_last_sigmoid', type=str2bool, default=True)

    # Criterion
    parser.add_argument('--crit', type=str, nargs='+', default=["bcelogit"])  # for cls
    parser.add_argument('--crit_coef', type=float, nargs='+', default=[1.0])  # for cls
    parser.add_argument('--crit_aux', type=str, nargs='+', default=[])  # for img, tab
    parser.add_argument('--crit_aux_coef', type=float, nargs='+', default=[1.0, 1.0])  # for img, tab
    
    # Optimizer
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_min', type=float, default=1e-7)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--opt_param', type=str, default="baseline")  # baseline / custom
    parser.add_argument('--opt_coef', type=float, nargs='+', default=[1.0, 1.0, 1.0])  # img tab cls

    # Scheduler
    parser.add_argument('--sched', type=str, default="cosine_warmup")
    parser.add_argument('--warmup_steps', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=1)

    args = parser.parse_args()
    args.base_dir = './'
    args.data_dir = osp.join(args.base_dir, 'data')
    return args


def main(args):
    args.device = torch.device("cuda:0")
    
    df = pd.read_csv(osp.join(args.data_dir, f'{args.df}.csv'))
    df = preprocess_df(df, args.df_ver, drop_row=True)

    if args.fold == 0:
        train_df = df.drop(columns=['kfold']).reset_index(drop=True)
        valid_loader = None
    else:
        train_df = df[df["kfold"] != args.fold].drop(columns=['kfold']).reset_index(drop=True)
        valid_df = df[df["kfold"] == args.fold].drop(columns=['kfold']).reset_index(drop=True)
        valid_loader = create_data_loader(
            valid_df, 'valid', args.img_size, args.batch_size, args.num_workers, args.data_dir, norm_type=args.norm_type,
        )
    train_loader = create_data_loader(
        train_df, 'train', args.img_size, args.batch_size, args.num_workers, args.data_dir, norm_type=args.norm_type
    )
    
    model = ClassificationModel(args).to(args.device)
    
    criterion = []
    for crit in args.crit:
        if crit == 'bcelogit':
            criterion.append(nn.BCEWithLogitsLoss().to(args.device))
        elif crit == 'bce':
            criterion.append(nn.BCELoss().to(args.device))
        elif crit == 'l1':
            criterion.append(nn.L1Loss().to(args.device))
    assert len(criterion) == len(args.crit_coef)

    crit_aux = []
    for crit in args.crit_aux:
        if crit == 'bcelogit':
            crit_aux.append(nn.BCEWithLogitsLoss().to(args.device))
        elif crit == 'bce':
            crit_aux.append(nn.BCELoss().to(args.device))
        elif crit == 'l1':
            crit_aux.append(nn.L1Loss().to(args.device))
    
    if args.opt_param == 'baseline':
        params = model.parameters()
    # not working
    elif args.opt_param == 'custom':
        params = [
            {"params": model.img_feature_extractor.parameters(), "lr": args.lr*args.opt_coef[0]},
            {"params": model.tab_feature_extractor.parameters(), "lr": args.lr*args.opt_coef[1]},
            {"params": model.classifier.parameters(), "lr": args.lr*args.opt_coef[2]},
        ]
    
    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamp':
        optimizer = AdamP(params, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.sched == 'reduce':
        factor = args.gamma
        patience = args.patience
        min_lr = args.lr_min
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience, threshold_mode='abs', min_lr=min_lr, verbose=True)
    elif args.sched == 'cosine':
        T_0 = args.epochs
        eta_min = args.lr_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=eta_min, verbose=True)
    elif args.sched == 'cosine_warmup':
        first_cycle_steps = args.epochs
        max_lr = args.lr
        min_lr = args.lr_min
        warmup_steps = args.warmup_steps
        gamma = args.gamma
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=1.0,
            max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=gamma)
    
    run_train(model, train_loader, valid_loader, criterion, crit_aux, optimizer, scheduler, args)


if __name__ == '__main__':
    args = get_parser()
    fold_list = args.fold
    for fold in fold_list:
        args = get_parser()
        args.fold = fold
        args.work_dir = get_exp_dir(args.work_dir)
        args.config_dir = osp.join(args.work_dir, 'config.yaml')
        
        save_config(args, args.config_dir)
        set_seeds(args.seed)
        wandb_init(args)
        main(args)