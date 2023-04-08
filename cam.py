from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from model import ImgFeatureExtractor
from utils import load_config


def load_model(model_path, args):
    '''
    model 불러오기
    '''
    model = ImgFeatureExtractor(args)
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for n, v in state_dict.items():
        new_n = n.replace('img_feature_extractor.','')
        new_state_dict[new_n] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dir = "./work_dirs/exp27"

    args_path = os.path.join(dir, 'config.yaml')
    args = load_config(args_path)

    try:
        print(args.pretrained)
    except:
        args.pretrained = True

    model_path = os.path.join(dir, 'best.pt')
    model = load_model(model_path, args).to(device)
    model.eval()

    # select your model's target layer 
    target_layers = [model.backbone.features[-1][0]]

    df = pd.read_csv('./data/train_5fold.csv')
    train_df = df[
        (df["kfold"] != args.fold) & (df['mask_path'] != '-'
    )].drop(columns=['kfold']).reset_index(drop=True)
    valid_df = df[
        (df["kfold"] == args.fold) & (df['mask_path'] != '-')
    ].drop(columns=['kfold']).reset_index(drop=True)

    # target, mode = train_df, 'train'
    target, mode = valid_df, 'valid'

    i = random.randint(0, len(target) - 1)
    img_path = target['img_path'].iloc[i]
    img_path = os.path.join('./data', img_path).replace('\\.','')
    
    img = cv2.imread(img_path, 1)[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    img = cv2.resize(img, dsize=(int(w/10), int(h/10)), interpolation=cv2.INTER_AREA)
    
    rgb_img = np.float32(img) / 255

    input_tensor = torch.permute(
        torch.unsqueeze(torch.Tensor(rgb_img).to(device), 0), (0, 3, 1, 2)
    )

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
    )
    visualization_save = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    plt.matshow(visualization)
    exp = dir.split('/')[-1]
    plt.title(f'[img:{img_path}]  [{exp}]  [fold:{args.fold}]  [mode:{mode}]')
    plt.show()
    # cv2.imwrite("example.png", visualization_save)
