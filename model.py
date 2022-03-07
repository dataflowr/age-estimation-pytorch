import torch.nn as nn
import torch
import pretrainedmodels
import pretrainedmodels.utils
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
            ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        #img = np.array(img.cpu()).astype(np.uint8)
        img = self.aug.augment_image(img)
        return img

class AE_model(nn.Module):
    def __init__(self, model_name="se_resnext50_32x4d", age_range=75, pretrained="imagenet", emb_dim = 512, hidden_dim = 64, seq_len=8):
        super(AE_model, self).__init__()
        self.transform = ImgAugTransform()
        self.seq_len = seq_len
        self.age_range = age_range
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.CNN = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        self.num_features = self.CNN.last_linear.in_features
        self.CNN.last_linear = nn.Linear(in_features = self.num_features, out_features=emb_dim)
        self.classifier = nn.Linear(in_features = emb_dim, out_features=age_range)
        regressors = [nn.Linear(in_features = emb_dim, out_features=1) for i in range(age_range)]
        self.regressors = nn.Sequential(*regressors)
        # le bloc d'attention
        '''self.Queries = nn.Linear(emb_dim, hidden_dim)
        self.Keys = nn.Linear(emb_dim, hidden_dim)
        self.Values = nn.Linear(emb_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first = True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"'''


    def forward(self, x):
        # le codage par l'attention
        '''shape = x.shape
        #self.h0 = torch.randn(shape[0], self.hidden_dim).to(self.device)
        seq= torch.zeros((shape[0], self.seq_len, self.emb_dim)).to(self.device)
        index = 0
        for input_imgs in x:
            for i in range(self.seq_len):
                img_aug = input_imgs[i].unsqueeze(0)
                emb = self.CNN(img_aug).squeeze(0)
                seq[index,i] = emb
            index+=1
        Q = self.Queries(seq) # (b,seq_len, emb_dim)
        K = self.Keys(seq) #//
        V = self.Values(seq) #torch.Size([b, seq_len, emb_dim])
        A = torch.einsum('bld,bsd->bls', seq,seq) # (b, seq_len, seq_len)
        A = F.softmax(A, dim=1)
        y = torch.einsum('bsl, bsd->bld',A,seq) # (b, seq_len, emb_dim)
        out, h = self.gru(y)
        #print(y.shape)
        embbeding = h[-1,:,:]
        #embbeding = torch.sum(y, dim=1)'''

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

