import torch
import torch.nn as nn
import torchvision.models as models


class ImgFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(ImgFeatureExtractor, self).__init__()
        if model == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.embedding = nn.Linear(1000, 512)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x


class TabFeatureExtractor(nn.Module):
    def __init__(self, model, init_feat):
        super(TabFeatureExtractor, self).__init__()
        if model == 'baseline':
            self.embedding = nn.Sequential(
                nn.Linear(in_features=init_feat, out_features=128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(in_features=128, out_features=256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(in_features=512, out_features=512)
            )
        
    def forward(self, x):
        x = self.embedding(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        self.img_feature_extractor = ImgFeatureExtractor(args.img_model)
        self.tab_feature_extractor = TabFeatureExtractor(args.tab_model, args.tab_init_feat)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid(),
        )
        
    def forward(self, img, tabular):
        img_feature = self.img_feature_extractor(img)
        tab_feature = self.tab_feature_extractor(tabular)
        feature = torch.cat([img_feature, tab_feature], dim=-1)
        output = self.classifier(feature)
        return output