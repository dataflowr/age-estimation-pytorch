import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils

class Homosedastic(nn.Module):

    def __init__(self, model_name='vgg13'):
        super(Homosedastic, self).__init__()

        self.common =  pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        print("original model",model)
        dim_feats = model.last_linear.in_features
        #The homosedastic model outputs 1 element (the age)
        model.last_linear = nn.Sequential(nn.Linear(dim_feats,101),nn.Linear(101, 40) ,nn.Linear(40,1), nn.ReLU())
        #model.avg_pool = nn.Identity()
        print("new model",model)
        
        self.hom_error = nn.Parameter(torch.zeros(1).float())

    def forward(self, x):
        return self.common(x)
    
def get_model(model_name="vgg13", num_classes=101, pretrained="imagenet",homosedastic=False):
    
    if not homosedastic:
        model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        print("original model",model)
        dim_feats = model.last_linear.in_features
        #The heterscedastic model outputs 2 elements (the age, the error)
        model.last_linear = nn.Sequential(nn.Linear(dim_feats,101),nn.Linear(101, 40) ,nn.Linear(40,2), nn.ReLU())
        #model.avg_pool = nn.Identity()
        print("new model",model)
        return model
    else:
        return Homosedastic(model_name)
      


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
