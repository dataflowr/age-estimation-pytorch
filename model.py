import torch.nn as nn
import torch
import pretrainedmodels
import pretrainedmodels.utils
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


class AE_model(nn.Module):
    def __init__(self, model_name="se_resnext50_32x4d", age_range=75, pretrained="imagenet", emb_dim = 512):
        super(AE_model, self).__init__()
        self.age_range = age_range
        self.CNN = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        self.num_features = self.CNN.last_linear.in_features
        self.CNN.last_linear = nn.Linear(in_features = self.num_features, out_features=emb_dim)
        self.classifier = nn.Linear(in_features = emb_dim, out_features=age_range)
        regressors = [nn.Linear(in_features = emb_dim, out_features=1) for i in range(age_range)]
        self.regressors = nn.Sequential(*regressors)

    def forward(self, x):
        embbeding = self.CNN(x)
        prob = F.softmax(self.classifier(embbeding), dim =1)
        pred = torch.zeros_like(prob)
        for i, reg in enumerate(self.regressors):
            pred[:,i] = reg(embbeding).squeeze(1)
        return prob, pred


def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = AE_model()
    return model

def main():
    model = AE_model()
    print(model)


if __name__ == '__main__':
    main()

