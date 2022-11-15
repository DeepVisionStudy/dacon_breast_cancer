import torch
import torch.nn as nn
import torchvision.models as models


class ImgFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(ImgFeatureExtractor, self).__init__()
        self.args = args
        # only for torchvision models
        self.backbone = getattr(models, self.args.img_model)(weights=self.get_weights())
        self.change_last_layer()
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def get_weights(self):
        if self.args.img_model == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        elif self.args.img_model == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        elif self.args.img_model == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        elif self.args.img_model == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        elif self.args.img_model =='swin_t':
            weights = models.Swin_T_Weights.IMAGENET1K_V1
        return weights
    
    def change_last_layer(self):
        in_features = 0
        for name, item in self.backbone.named_children():
            if 'fc' in name:
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features, self.args.img_last_feat)
            if 'classifier' in name:
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, self.args.img_last_feat)
            if 'head' in name:
                if 'swin' in self.args.img_model:
                    in_features = self.backbone.head.in_features
                    self.backbone.head = nn.Linear(in_features, self.args.img_last_feat)
                elif 'vit' in self.args.img_model:
                    in_features = self.backbone.heads.head.in_features
                    self.backbone.heads = nn.Linear(in_features, self.args.img_last_feat)
            
            if in_features != 0:
                break


class TabFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(TabFeatureExtractor, self).__init__()
        if args.tab_model == 'baseline':
            self.embedding = nn.Sequential(
                nn.Linear(in_features=args.tab_init_feat, out_features=128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(in_features=128, out_features=256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(in_features=512, out_features=args.tab_last_feat)
            )
        
    def forward(self, x):
        x = self.embedding(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        self.img_feature_extractor = ImgFeatureExtractor(args)
        self.tab_feature_extractor = TabFeatureExtractor(args)
        
        in_features = args.img_last_feat + args.tab_last_feat
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid(),
        )
        
    def forward(self, img, tabular):
        img_feature = self.img_feature_extractor(img)
        tab_feature = self.tab_feature_extractor(tabular)
        feature = torch.cat([img_feature, tab_feature], dim=-1)
        output = self.classifier(feature)
        return output


if __name__ == '__main__':
    model = models.convnext_tiny()
    print(model.named_modules)