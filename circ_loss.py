import torch
from torch import nn

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda') if CUDA else torch.device('cpu')


class CIRCLoss(nn.Module):

    def __init__(self, config):

        super(CIRCLoss, self).__init__()

        self.range = config.max_range - config.min_range
        self.mse = nn.MSELoss(reduction='none')
        self.min_dim = -1

    def forward(self, x, y):
        loss = torch.cat((self.mse(x, y), self.mse(x-self.range, y), self.mse(x+self.range, y)), self.min_dim)
        loss, ind = torch.min(loss, self.min_dim)
        loss = torch.mean(loss)

        return loss
