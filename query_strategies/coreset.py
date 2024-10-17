import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import Sampler

import numpy
import numpy as np
import math
import os
import time
import torchvision
import pickle
import cv2
from PIL import Image
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon

#from model.deeplabv2 import Res_Deeplab
from utils.loss import CrossEntropy2d
from data.dataloaders.pascal_voc import VOCDataSet, VOCGTDataSet, VOCOracleGTDataSet
from .strategy import Strategy
from data.dataloaders.a2d2 import A2D2
from sklearn.metrics import pairwise_distances

device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class CoreSet(Strategy):
    def __init__(self, list_train_imgs, idxs_lb, idxs_unlb, net, args, writer):
        super(CoreSet, self).__init__(list_train_imgs, idxs_lb, idxs_unlb, net, args, writer)

    def get_embeddings(self):
        self.net.eval()
        self.net.cuda()
        
        h, w = map(int, self.args.input_size.split(','))
        input_size = (h, w) 
        
        if self.args.dataset == 'pascal_voc':
            train_dataset = VOCDataSet(self.args.data_dir, self.args.train_data_list, crop_size=input_size, scale=self.args.random_scale, mirror=self.args.random_mirror, mean=IMG_MEAN)
        elif self.args.dataset == 'a2d2':
            train_dataset = A2D2(task='train', hflip=False, is_crop=True, pool_type='lab', lab_percent=self.args.lab_percent, pool=self.args.pool, split=self.args.split)
    
        trainloader = data.DataLoader(train_dataset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        total_cost = 0
        embeddings = []
         
        with torch.no_grad(): 
            for ind, batch in enumerate(trainloader):
                image, gt, size, name, index = batch
                #size = size[0] 
               
                output_feat  = self.net(image.cuda(), feat=True)

                if ind%500==0:
                    print ('getting embedding: ', ind, index)
            
                embeddings.append(output_feat[0].cpu().detach().numpy()) 

        lb_embeddings = [embeddings[item] for item in self.idxs_lb] 
        unlb_embeddings = [embeddings[item] for item in self.idxs_unlb] 
        return np.array(lb_embeddings), np.array(unlb_embeddings)
  
    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
        
        print (idxs)
        return idxs
 
    def query_and_update(self):
        new_samples = int(len(self.list_train_imgs)*self.args.sampling_image_budget)
        lb_embed, unlb_embed = self.get_embeddings()
        chosen = self.furthest_first(unlb_embed, lb_embed, new_samples)
        self.idxs_lb = np.concatenate((self.idxs_lb, self.idxs_unlb[chosen]), axis=0)
        self.idxs_unlb = [item for item in self.idxs_unlb if item not in self.idxs_lb]
        self.idxs_unlb = np.array(self.idxs_unlb) 
