import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transF
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import Sampler

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class EntropyImage(Strategy):
    def __init__(self, list_train_imgs, idxs_lb, idxs_unlb, net, args, writer):
        super(EntropyImage, self).__init__(list_train_imgs, idxs_lb, idxs_unlb, net, args, writer)

    def entropy_image(self, pred):
        b = F.softmax(pred, dim=0)*F.log_softmax(pred, dim=0)
        b = -1.0 * b.sum(dim=0)
        b = (255*(b - b.min())/(b.max()-b.min()))
        
        c = b.cpu().numpy().astype(np.uint8)
        out = cv2.applyColorMap(c, cv2.COLORMAP_JET)
        return out, torch.sum(b[b>150]).cpu().numpy()
       
    def entropy_image_avg(self, pred):
        b = F.softmax(pred, dim=0)*F.log_softmax(pred, dim=0)
        b = -1.0 * b.sum(dim=0)
        b = (255*(b - b.min())/(b.max()-b.min()))
        
        c = b.cpu().numpy().astype(np.uint8)
        out = cv2.applyColorMap(c, cv2.COLORMAP_JET)
        return out, torch.sum(b).cpu().numpy()/(b.size(0)*b.size(1))

        
    def sort_images_by_entropy(self):
        self.net.eval()
        self.net.cuda()
       
        h, w = map(int, self.args.input_size.split(','))
        input_size = (h, w)
 
        if self.args.dataset == 'pascal_voc':
            train_dataset = VOCDataSet(self.args.data_dir, self.args.train_data_list, crop_size=input_size, scale=False, mirror=False, mean=IMG_MEAN)
        elif self.args.dataset == 'a2d2':
            train_dataset = A2D2(task='train', hflip=False, is_crop=True, pool_type='lab', lab_percent=self.args.lab_percent, pool=self.args.pool, split=self.args.split)
    
        train_sampler = data.sampler.SubsetRandomSampler(self.idxs_unlb)
        trainloader = data.DataLoader(train_dataset, batch_size=self.args.test_batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        
        interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True) 
        total_cost = 0
        
        list_index_ent = []
         
        with torch.no_grad(): 
            for ind, batch in enumerate(trainloader):
                image, gt, size, name, index = batch
                size = size[0] 
                image_flip = transF.hflip(image)
                    
                output  = self.net(image.cuda())
                output = interp(output).squeeze(0)
                output = output[:,:size[0],:size[1]]

                entropy_heatmap, ent = self.entropy_image_avg(output)
                
                if ind%500==0:
                    print (ind)
            
                list_index_ent.append((index, ent))
        
        list_index_ent = [(tensor.numpy(), float(value)) for tensor, value in list_index_ent]
        list_index_ent.sort(key=lambda x: x[1], reverse=True)
        # for item in list_index_ent:
        #     if len(item) != 2:
        #         print("Inconsistent item:", item)
        # import pdb; pdb.set_trace()

        list_index_ent = np.array(list_index_ent, dtype=object)
        print ('sorted list index ent: ', list_index_ent[:10])

        sorted_idxs = list_index_ent[:,0]
        print(sorted_idxs.shape)
 
        return sorted_idxs
   
    def query_and_update(self):
        sorted_idxs = self.sort_images_by_entropy()
        sorted_idxs = [int(item) for item in sorted_idxs]
        new_samples = int(len(self.list_train_imgs)*self.args.sampling_image_budget)
        print ('New images:', sorted_idxs[:new_samples])
        self.idxs_lb  = np.concatenate((self.idxs_lb, sorted_idxs[:new_samples]), axis=0)
        self.idxs_unlb = [item for item in self.idxs_unlb if item not in self.idxs_lb]     
 
