import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils

class Homosedastic(nn.Module):

    def __init__(self):
        super(Homosedastic, self).__init__()

        self.common = get_model()
        self.hom_error = nn.Parameter(torch.zeros(1).float())

    def forward(self, x):
        return self.common(x)
    
def get_model(model_name="vgg13", num_classes=101, pretrained="imagenet",homosedastic=False):
  
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    print("original model",model)
    dim_feats = model.last_linear.in_features
    if not homosedastic:
        model.last_linear = nn.Sequential(nn.Linear(dim_feats,101),nn.Linear(101, 40) ,nn.Linear(40,2), nn.ReLU())
    #model.avg_pool = nn.Identity()
    print("new model",model)
    return model
      


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
