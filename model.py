import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils


def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def get_classifier(n_features, n_classes=101):
    hidden = int(n_features/10)
    classif = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes),
        # nn.Softmax()
        # TODO: try a Softmax or normalization
    )
    return classif

def get_regressor(n_features, n_classes):
    hidden = int(n_features/10)
    reg = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden , n_classes),
        nn.ReLU(),
        nn.Linear(n_classes, 1)
    )
    return reg

def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
