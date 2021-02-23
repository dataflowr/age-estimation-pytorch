from model import get_model
from dataset import FaceDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from defaults import _C as cfg
from train import get_args
import torch

def preconvfeat(dataloader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    conv_features = []
    labels_list = []
    for data in tqdm(dataloader):
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        x = model.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return (conv_features,labels_list)


def main():
    args = get_args()

    model = get_model()
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    features, labels = preconvfeat(train_loader, model)
    print(features.shape, len(labels))


if __name__ == '__main__':
    main()