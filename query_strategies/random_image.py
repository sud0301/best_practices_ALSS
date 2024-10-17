import torch
import torch.nn as nn

import numpy as np
import math
import os
import torchvision
from .strategy import Strategy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class RandomImage(Strategy):
    def __init__(self, list_train_imgs, idxs_lb, idxs_unlb, net, args, writer):
        super().__init__(list_train_imgs, idxs_lb, idxs_unlb, net, args, writer)

    def query_and_update(self):
        new_samples = int(len(self.list_train_imgs)*self.args.sampling_image_budget)
        print ('new samples: ', self.idxs_unlb[:new_samples])
        self.idxs_lb  = np.concatenate((self.idxs_lb, self.idxs_unlb[:new_samples]), axis=0)
        self.idxs_unlb = [item for item in self.idxs_unlb if item not in self.idxs_lb]
