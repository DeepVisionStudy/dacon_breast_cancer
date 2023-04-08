import os.path as osp

import torch
import torch.nn as nn
import torchvision.models as models

import edgenext_model.models.model as edgenext_models


class ImgFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(ImgFeatureExtractor, self).__init__()
        self.args = args
        
        if 'edgenext' in args.img_model:
            self.backbone = getattr(edgenext_models, args.img_model)(pretrained=self.args.pretrained, classifier_dropout=0.0)
            self.backbone.head = nn.Sequential(nn.Linear(self.backbone.head.in_features, self.args.img_last_feat))
        else:
            self.backbone = getattr(models, args.img_model)(weights=self.get_weights())
            self.change_last_layer()

    def forward(self, x):
        x = self.backbone(x)
        return x

    def get_weights(self):
        if not self.args.pretrained:
            weights = None
        else:
            if self.args.img_model == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        return weights
    
    def change_last_layer(self):
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, self.args.img_last_feat)
    
    def requires_grad(self, phase, boolean=True):
        if phase == 'back':
            network = self.backbone
        elif phase == 'head':
            network = self.backbone.classifier

        for name, layer in network.named_children():
            for param in layer.parameters():
                param.requires_grad = boolean


class TabFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(TabFeatureExtractor, self).__init__()
        if args.tab_model == 'drop20':
            self.backbone = nn.Sequential(
                nn.Linear(args.tab_init_feat, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(256, 512),
            )
            self.embedding = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, args.tab_last_feat),
            )
        elif args.tab_model == 'add64feat':
            self.backbone = nn.Sequential(
                nn.Linear(args.tab_init_feat, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(256, 512),
            )
            self.embedding = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, args.tab_last_feat),
            )
        elif args.tab_model == 'drop2':
            self.backbone = nn.Sequential(
                nn.Linear(args.tab_init_feat, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(256, 512),
            )
            self.embedding = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, args.tab_last_feat),
            )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x
    
    def requires_grad(self, phase, boolean=True):
        if phase == 'back':
            network = self.backbone
        elif phase == 'head':
            network = self.embedding

        for name, layer in network.named_children():
            for param in layer.parameters():
                param.requires_grad = boolean


class ClassificationModel(nn.Module):
    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        self.args = args

        if args.cls_model == 'baseline':
            self.img_feature_extractor = ImgFeatureExtractor(args)
            self.tab_feature_extractor = TabFeatureExtractor(args)
        elif args.cls_model == 'img2tab':
            args.tab_init_feat += 1
            self.img_feature_extractor = ImgFeatureExtractor(args)
            self.tab_feature_extractor = TabFeatureExtractor(args)
        
        if args.cls_fusion == 'cat':
            in_features = args.img_last_feat + args.tab_last_feat
            layers = [nn.Linear(in_features=in_features, out_features=1)]
        if args.cls_last_sigmoid:
            layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, img, tab):
        if self.args.cls_model == 'baseline':
            img_feature = self.img_feature_extractor(img)
            tab_feature = self.tab_feature_extractor(tab)
        elif self.args.cls_model == 'img2tab':
            img_feature = self.img_feature_extractor(img)
            img2tab_feature = torch.cat([img_feature, tab], dim=-1)
            tab_feature = self.tab_feature_extractor(img2tab_feature)

        if self.args.cls_fusion == 'cat':
            feature = torch.cat([img_feature, tab_feature], dim=-1)
            output = self.classifier(feature)
        
        return output

if __name__ == '__main__':
    model = models.efficientnet_b0()
    for name, item in model.named_children():
        if 'fc' in name:
            print(name)
        if 'classifier' in name:
            print(name)
        if 'head' in name:
            print(name)
    
    print(model)