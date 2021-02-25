from model import get_model
from dataset import FaceDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from defaults import _C as cfg
from train import get_args
import torch
import pandas as pd
import pickle
import torch.nn.functional as F

def preconvfeat(dataloader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    conv_features = []
    labels_list = []
    for data in tqdm(dataloader):
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        x = F.adaptive_avg_pool2d(model.features(inputs),1)   
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return (conv_features,labels_list)


def get_feature_loader(features, labels, batch_size, shuffle, num_workers, drop_last):
    dtype = torch.float
    datasetfeat = [[torch.from_numpy(f).type(dtype),torch.tensor(l).type(torch.long)] for (f,l) in zip(features,labels)]
    datasetfeat = [(inputs.reshape(-1), classes) for [inputs,classes] in datasetfeat]
    loaderfeat = DataLoader(datasetfeat, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return loaderfeat


def main():
    args = get_args()

    model = get_model()
    
    # precompute validation Features
    valid_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    valid_features, valid_labels = preconvfeat(valid_loader, model)

    with open('valid_features.pkl','wb') as f:
        pickle.dump(valid_features, f)
    
    with open('valid_labels.pkl','wb') as f:
        pickle.dump(valid_labels, f)


    # precompute training Features
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    train_features, train_labels = preconvfeat(train_loader, model)

    with open('train_features.pkl','wb') as f:
        pickle.dump(train_features, f)
    with open('train_labels.pkl','wb') as f:
        pickle.dump(train_labels, f)

    

if __name__ == '__main__':
    main()