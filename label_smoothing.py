import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, std_smoothing, n_classes):
        assert 0.0 < std_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.std = std_smoothing * n_classes
        self.x = np.arange(0, n_classes, 1)
        # self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # model_prob = np.array([])
        # for age in target:
        model_prob = norm.pdf(self.x, target.unsqueeze(1), self.std)
        
        model_prob = torch.from_numpy(model_prob)
        print(model_prob.shape)

        return F.kl_div(output, model_prob, reduction='sum')

def main():
    output = torch.zeros(5,10)
    target = torch.Tensor([4,6,2,9,6])

    criterion = LabelSmoothingLoss(0.05, 10)
    loss = criterion(output, target)

if __name__ == '__main__':
    main()

