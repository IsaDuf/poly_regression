import torch
from torch import nn
import math

torch.manual_seed(1023210)


class Classifier(nn.Module):

    def __init__(self, config, input_shape):

        super(Classifier, self).__init__()

        self.config = config
        self.feat = config.feat

        self.min_range = config.min_range
        self.max_range = config.max_range

        self.fc = nn.Linear(input_shape[-1], config.num_class)

    def forward(self, x):
        x = self.fc(x)

        return x


class PWCirc(nn.Module):

    def __init__(self, config, input_shape):

        super(PWCirc, self).__init__()

        self.circ_sync = config.circ_sync or config.upper_level == 'circ_sync'
        self.num_cuts = config.num_cuts
        self.feat = config.feat

        self.min_range = config.min_range
        self.max_range = config.max_range
        self.range = self.max_range - self.min_range

        # use group convolution with kernel size = 1 instead of multiple linear models for efficiency
        # (parallel training)
        # random state is saved to allow deterministic weight initialization
        # and replicate the previous module_list implementation
        this_state = torch.get_rng_state()
        self.multi_head_linear = nn.Conv1d(input_shape[-1] * self.num_cuts,
                                           self.num_cuts, kernel_size=1, groups=self.num_cuts)
        torch.set_rng_state(this_state)

        # use default init for linear weights instead of convolution
        bound = 1 / math.sqrt(self.multi_head_linear.weight.size(1))

        for i in range(self.multi_head_linear.weight.size(0)):
            torch.nn.init.uniform_(self.multi_head_linear.weight[i], -bound, bound)
            if self.multi_head_linear.bias is not None:
                torch.nn.init.uniform_(self.multi_head_linear.bias[i], -bound, bound)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1, self.num_cuts, 1)
        x = self.multi_head_linear(x)

        return x
