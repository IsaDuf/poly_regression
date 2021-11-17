import numpy as np
import torch
from torch import nn, tensor
from scipy.stats import circmean


def ptocirc(logits, num_class, high=6, low=1):
    logits = logits
    bound = num_class//2

    # get probabilities
    soft = nn.Softmax(dim=1)
    p = soft(logits).detach().cpu().numpy()

    # find class with highest probability
    max = np.argmax(p, axis=1)

    # TODO fix for even number of class
    # generate circular classes values from probability
    # by placing highest value in the middle and rotate
    if num_class%2 != 0:
        circ = np.array([np.roll(np.arange(max[i] - bound, float(max[i] + bound + 1)),
                                 max[i] - (bound + 1))
                         for i in range(len(logits))])

    else:
        circ = np.array([np.roll(np.arange(max[i] - bound, float(max[i] + bound)),
                                 max[i] - (bound + 1))
                         for i in range(len(logits))])

    # get circular predictions and bound output to circular model
    circ_pred = circ*p
    circ_pred = np.sum(circ_pred, axis=1)
    circ_pred = np.vstack((circ_pred, circ_pred))
    circ_pred = circmean(circ_pred, high=high, low=low, axis=0)

    return circ_pred
