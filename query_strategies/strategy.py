import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import math
import os
import time
import torchvision
import pickle
import cv2
from PIL import Image

#from model.deeplabv2 import Res_Deeplab
from utils.loss import CrossEntropy2d #, CrossEntropy2d_LL
from utils.misc import AverageMeter, VOCColorize
from utils.metric import get_iou
from data.dataloaders.pascal_voc import VOCDataSet, VOCGTDataSet #, VOCPolyGTDataSet, VOCOracleGTDataSet
from data.dataloaders.a2d2 import A2D2
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Strategy:
    def __init__(self, list_train_imgs, idxs_lb, idxs_unlb, net, args, writer):
        self.list_train_imgs = list_train_imgs
        self.idxs_lb = idxs_lb
        self.idxs_unlb = idxs_unlb
        self.net = net
        self.args = args
        self.writer = writer
        torch.manual_seed(self.args.seed)
    
    def query_and_update(self):
        pass

    def loss_calc(self, pred, label, gpu):
        label = label.long().cuda()
        criterion = CrossEntropy2d(ignore_label=self.args.ignore_label).cuda()
        return criterion(pred, label)

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))

    def adjust_learning_rate(self, optimizer, i_iter):
        lr = self.lr_poly(self.args.learning_rate, i_iter, self.args.num_steps, self.args.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1 :
            optimizer.param_groups[1]['lr'] = lr * 10
  
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
     
    def train(self, rd):
        loss_value = AverageMeter()
        h, w = map(int, self.args.input_size.split(','))
        input_size = (h, w)
        
        cudnn.enabled = True
        #self.net = self.net()
        self.clf = self.net 
        #self.clf.apply(self.weights_init)

        saved_state_dict = torch.load(self.args.restore_from)['state_dict']
        saved_state_dict = {k.replace("module.", "backbone."): v for k, v in saved_state_dict.items()}
        # only copy the params that exist in current model (caffe-like)
        #saved_state_dict = torch.load('./pretrained_models/resnet101-5d3b4d8f.pth')
        new_params = self.clf.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
                #print (name)
        
        self.clf.load_state_dict(new_params)

        self.clf = torch.nn.DataParallel(self.clf).cuda()
        self.clf.train()
        self.clf.cuda()
        
        cudnn.benchmark = True
       
        if self.args.dataset == 'pascal_voc': 
            train_dataset = VOCDataSet(self.args.data_dir, self.args.train_data_list, crop_size=input_size, scale=True, mirror=True, mean=IMG_MEAN)
        elif self.args.dataset == 'a2d2':
            train_dataset = A2D2(task='train', hflip=True, is_crop=True, pool_type='lab', lab_percent=self.args.lab_percent, pool=self.args.pool, split=self.args.split)
        
        train_sampler = data.sampler.SubsetRandomSampler(self.idxs_lb)
        trainloader = data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)

        #optimizer = optim.SGD(self.net.module.optim_parameters(self.args), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        optimizer = optim.SGD(self.clf.module.optim_parameters(self.args), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
         
        trainloader_iter = iter(trainloader)
  
        best_miou = 0
        full_miou = 0
 
        for i_iter in range(self.args.num_steps):
            self.clf.train()
            optimizer.zero_grad()
            self.adjust_learning_rate(optimizer, i_iter)

            try:
                batch_lab = next(trainloader_iter)
            except:
                trainloader_iter = iter(trainloader)
                batch_lab = next(trainloader_iter)

            images, labels, _, _, index = batch_lab
            images = images.cuda()

            output = self.clf(images, montecarlo=self.args.final_dropout)
            pred = interp(output)
            loss_ce = self.loss_calc(pred, labels, self.args.gpu)
            loss = loss_ce
        
            loss.backward()
            loss_value.update(loss.item())

            optimizer.step()

            if i_iter%100==0:
                print('iter = {0:5d}/{1:5d}, loss_seg = {2:.3f}'.format(i_iter, self.args.num_steps, loss_value.avg))
                loss_value.reset()
        
                        
            if (i_iter % self.args.save_pred_every == 0 and i_iter!=0):
                miou = self.eval(samples='hundred')
                if miou > best_miou:
                    best_miou = miou
                    print ('saving best checkpoint! Best miou: ', best_miou)
                    full_miou = self.eval()
                    torch.save(self.clf.state_dict(), os.path.join(self.args.checkpoint_dir, 'VOC_round_' + str(rd) + '_' +str(i_iter)+'.pth'))
            
            if i_iter == self.args.num_steps-1:
                print ('saving last checkpoint!')
                full_miou = self.eval()
                torch.save({
                    'net_state_dict': self.clf.module.state_dict(),
                    'idxs_lb': self.idxs_lb,
                    'idxs_unlb': self.idxs_unlb,
                    'rd_idx': rd}, os.path.join(self.args.checkpoint_dir, 'SL_' + str(rd) + '_' + str(i_iter) + '.pth'))           
 
        return full_miou
    
    def eval(self, samples=None):
        self.net.eval()
        self.net.cuda()
   
        if self.args.dataset == 'a2d2': 
            val_dataset = A2D2(transform=False, task='val', hflip=False, is_crop=False) 
            valloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)
            interp = nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)
        elif self.args.dataset == 'pascal_voc':
            valloader = data.DataLoader(VOCGTDataSet(self.args.data_dir, self.args.test_data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False, mirror=False),
                                 batch_size=self.args.test_batch_size, shuffle=False, pin_memory=True)
            interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True) 
    
        data_list = []
        #colorize = VOCColorize() 

        with torch.no_grad(): 
            for index, batch in enumerate(valloader):
                if index % 100 == 0 and index !=0:
                    print('%d processd'%(index))
                    if samples == 'hundred':
                        break
                image, label, size, name, _ = batch
                size = size[0]
                output  = self.net(image.cuda(), montecarlo=self.args.final_dropout)
                output = interp(output).cpu().data[0].numpy()

                output = output[:,:size[0],:size[1]]
                gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int32)

                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
                
                #filename = os.path.join(self.args.save_dir, '{}.png'.format(name[0]))
                #color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
                #color_file.save(filename)

                data_list.append([gt.flatten(), output.flatten()])
        return get_iou(self.args, data_list)
