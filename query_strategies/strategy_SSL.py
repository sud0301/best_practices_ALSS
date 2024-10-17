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
from utils.misc import AverageMeter

from utils.loss import CrossEntropy2d 
from utils.misc import AverageMeter, VOCColorize
from utils.metric import get_iou
from data.dataloaders.pascal_voc import VOCDataSet, VOCGTDataSet 
from data.dataloaders.a2d2 import A2D2
#from data.brain_slices_dataset import Brain_SliceGTDataSet, Brain_Slice_Set

from model.deeplabv3p import DeepV3PlusW38
from model.discriminator import S4GAN_D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class StrategySSL:
    def __init__(self, list_train_imgs, idxs_lb, idxs_unlb, net, net_D, args, writer):
       
        self.list_train_imgs = list_train_imgs
        self.idxs_lb = idxs_lb
        self.idxs_lb_full = idxs_lb
        self.idxs_unlb = idxs_unlb
        self.net = net
        self.net_D = net_D
        self.args = args
        torch.manual_seed(self.args.seed)
    
    def query_and_update(self):
        pass

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def loss_calc(self, pred, label, gpu):
        label = label.long().cuda()
        criterion = CrossEntropy2d(ignore_label=self.args.ignore_label).cuda()
        return criterion(pred, label)

    def loss_calc_st(self, pred, label, gpu):
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
   
    def adjust_learning_rate_D(self, optimizer, i_iter):
        lr = self.lr_poly(self.args.learning_rate_D, i_iter, self.args.num_steps, self.args.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1 :
            optimizer.param_groups[1]['lr'] = lr * 10

    def one_hot_ssl(self, label):
        label = label.numpy()
        one_hot = np.zeros((label.shape[0], self.args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
        for i in range(self.args.num_classes):
            one_hot[:,i,...] = (label==i)
        #handle ignore labels
        return torch.FloatTensor(one_hot)
 
    def compute_argmax_map(self, output):
        output = output.detach().cpu().numpy()
        output = output.transpose((1,2,0))
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
        output = torch.from_numpy(output).float()
        return output
 
    def find_good_maps(self, D_outs, pred_all):
        count = 0
        for i in range(D_outs.size(0)):
            if D_outs[i] > self.args.threshold_st:
                count +=1

        if count > 0:
            #print ('Above ST-Threshold : ', count, '/', self.args.train_batch_size)
            pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
            label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
            num_sel = 0
            for j in range(D_outs.size(0)):
                if D_outs[j] > self.args.threshold_st:
                    pred_sel[num_sel] = pred_all[j]
                    label_sel[num_sel] = self.compute_argmax_map(pred_all[j])
                    num_sel +=1
            return  pred_sel.cuda(), label_sel.cuda(), count
        else:
            return 0, 0, count 

    def get_class_cost(self, target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
        vect = hist>0
        class_cost = 0
        for i in range(len(vect)):
            if vect[i] == True:
                class_cost +=1
        return class_cost
           
    def train(self, rd):
        h, w = map(int, self.args.input_size.split(','))
        input_size = (h, w)
        
        cudnn.enabled = True
        self.clf = self.net #().cuda()
        self.clf_D = self.net_D #().cuda()
   
        self.clf.apply(self.weights_init)
        self.clf_D.apply(self.weights_init)

        #self.clf.apply(self.weights_init)
        saved_state_dict = torch.load(self.args.restore_from)['state_dict']
        # saved_state_dict = torch.load("/home/host/ki-deltalearning/ap_3_3/KIDL_active_learning/log/checkpoints/brain_slice_pool5_entropy_SSL_ep100/SSL_0_5011.pth")
        saved_state_dict = {k.replace("module.", "backbone."): v for k, v in saved_state_dict.items()}
        # only copy the params that exist in current model (caffe-like)
        new_params = self.clf.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
                #print (name)
        self.clf.load_state_dict(new_params)

        criterion = nn.BCELoss()

        self.clf = torch.nn.DataParallel(self.clf).cuda()
        self.clf.train()

        self.clf_D = torch.nn.DataParallel(self.clf_D).cuda()
        self.clf_D.train()
        
        cudnn.benchmark = True
   
        if self.args.dataset == 'pascal_voc':
            train_dataset = VOCDataSet(self.args.data_dir, self.args.train_data_list, crop_size=input_size, scale=True, mirror=True, mean=IMG_MEAN)
            train_dataset_gt = VOCGTDataSet(self.args.data_dir, self.args.train_data_list, crop_size=input_size, scale=True, mirror=True, mean=IMG_MEAN)
        # elif self.args.dataset == 'brain_slice':
        #     train_dataset = Brain_Slice_Set(self.args.data_dir, self.args.train_data_list, crop_size=input_size, scale=True, mirror=False, mean=IMG_MEAN)
        #     train_dataset_gt = Brain_SliceGTDataSet(self.args.data_dir, self.args.train_data_list, crop_size=input_size, scale=True, mirror=False, mean=IMG_MEAN)            
        elif self.args.dataset == 'a2d2':
            train_dataset = A2D2(task='train', hflip=True, is_crop=True, pool_type='lab', lab_percent=self.args.lab_percent, pool=self.args.pool, split=self.args.split)
            train_dataset_gt = A2D2(task='train', hflip=True, is_crop=True, pool_type='lab', lab_percent=self.args.lab_percent, pool=self.args.pool, split=self.args.split)
 
        train_sampler = data.sampler.SubsetRandomSampler(self.idxs_lb)
        train_remain_sampler = data.sampler.SubsetRandomSampler(self.idxs_unlb)
        train_gt_sampler = data.sampler.SubsetRandomSampler(self.idxs_lb)

        trainloader = data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, sampler=train_sampler, num_workers=2, pin_memory=False, drop_last=True)
        trainloader_remain = data.DataLoader(train_dataset, batch_size=self.args.train_batch_size, sampler=train_remain_sampler, num_workers=2, pin_memory=False, drop_last=True)
        trainloader_gt = data.DataLoader(train_dataset_gt, batch_size=self.args.train_batch_size, sampler=train_gt_sampler, num_workers=2, pin_memory=False, drop_last=True)

        trainloader_iter = iter(trainloader)
        trainloader_remain_iter = iter(trainloader_remain)
        trainloader_gt_iter = iter(trainloader_gt)

        # optimizer for segmentation network
        optimizer = optim.SGD(self.clf.module.optim_parameters(self.args), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        # optimizer for discriminator network
        optimizer_D = optim.Adam(self.clf_D.parameters(), lr=self.args.learning_rate_D, betas=(0.9,0.99))

        interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

        # labels for adversarial training
        pred_label = 0
        gt_label = 1

        y_real_, y_fake_ = torch.ones(self.args.train_batch_size, 1).cuda(), torch.zeros(self.args.train_batch_size, 1).cuda()

        loss_ce_value = AverageMeter()
        loss_D_value = AverageMeter()
        loss_fm_value = AverageMeter()
        loss_S_value = AverageMeter()
        counts = AverageMeter()

        best_miou = 0
        full_miou = 0

        print ('Training starts..')
        for i_iter in range(self.args.num_steps):
            ####
            # self.eval()
            # if i_iter%100==0:
            #     self.eval(samples='hundred')
            ####
            self.clf.train()
            optimizer.zero_grad()
            self.adjust_learning_rate(optimizer, i_iter)
            optimizer_D.zero_grad()
            self.adjust_learning_rate_D(optimizer_D, i_iter)

            # train Segmentation Network 
            # don't accumulate grads in D
            for param in self.clf_D.parameters():
                param.requires_grad = False

            # training loss for labeled data only
            try:
                batch = next(trainloader_iter)
            except:
                trainloader_iter = iter(trainloader)
                batch = next(trainloader_iter)
        
            images, labels, _, _, _ = batch
            # from torchvision.utils import save_image
            # save_image(images[0], 'img1.png') 
            # save_image(labels[0] * 100, 'label1.png') 
            # print(torch.unique(labels[0]))
            images = images.cuda()
            
            output = self.clf(images, montecarlo=self.args.final_dropout)
            pred = interp(output)
            # print(torch.unique(torch.argmax(pred[0],0)))
            loss_seg = self.loss_calc(pred, labels, self.args.gpu)
            loss_ce = loss_seg

            #training loss for remaining unlabeled data
            try:
                batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            images_remain, _, _, _, _ = batch_remain
            images_remain = images_remain.cuda()
            out_remain = self.clf(images_remain, montecarlo=self.args.final_dropout)
            pred_remain = interp(out_remain)

            # concatenate the prediction with the input images
            images_remain = (images_remain-torch.min(images_remain))/(torch.max(images_remain)- torch.min(images_remain))
            pred_cat = torch.cat((F.softmax(pred_remain, dim=1), images_remain), dim=1)

            D_out_z, D_out_y_pred = self.clf_D(pred_cat) # predicts the D ouput 0-1 and feature map for FM-loss 

            # find predicted segmentation maps above threshold 
            pred_sel, labels_sel, count = self.find_good_maps(D_out_z, pred_remain)
            counts.update(count)

            # training loss on above threshold segmentation predictions (Cross Entropy Loss)
            if count > 0 and i_iter > 0:
                loss_st = self.loss_calc_st(pred_sel, labels_sel, self.args.gpu)
            else:
                loss_st = 0.0

            # Concatenates the input images and ground-truth maps for the Districrimator 'Real' input
            try:
                batch_gt = next(trainloader_gt_iter)
            except:
                trainloader_gt_iter = iter(trainloader_gt)
                batch_gt = next(trainloader_gt_iter)

            images_gt, labels_gt, _, _, _ = batch_gt
            # Converts grounth truth segmentation into 'num_classes' segmentation maps.
            D_gt_v = Variable(self.one_hot_ssl(labels_gt)).cuda()

            images_gt = images_gt.cuda()
            images_gt = (images_gt - torch.min(images_gt))/(torch.max(images)-torch.min(images))

            D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim=1)
            D_out_z_gt , D_out_y_gt = self.clf_D(D_gt_v_cat)

            # L1 loss for Feature Matching Loss
            loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))

            if count > 0 and i_iter > 0: # if any good predictions found for self-training loss
                loss_S = loss_ce +  self.args.lambda_fm*loss_fm + self.args.lambda_st*loss_st
            else:
                loss_S = loss_ce + self.args.lambda_fm*loss_fm

            loss_S.backward()

            loss_fm_value.update(self.args.lambda_fm*loss_fm)
            loss_ce_value.update(loss_ce.item())
            loss_S_value.update(loss_S.item())
            
            # train D
            for param in self.clf_D.parameters():
                param.requires_grad = True

            # train with pred
            pred_cat = pred_cat.detach()  # detach does not allow the graddients to back propagate.

            D_out_z, _ = self.clf_D(pred_cat)
            y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
            loss_D_fake = criterion(D_out_z, y_fake_)

            # train with gt
            D_out_z_gt , _ = self.clf_D(D_gt_v_cat)
            y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda())
            loss_D_real = criterion(D_out_z_gt, y_real_)

            loss_D = (loss_D_fake + loss_D_real)/2.0
            loss_D.backward()
            loss_D_value.update(loss_D.item())

            optimizer.step()
            optimizer_D.step()

            if i_iter%100==0:
                print('iter = {0:8d}/{1:8d}, loss_ce = {2:.3f}, loss_fm = {3:.3f}, loss_S = {4:.3f}, loss_D = {5:.3f}, count = {6:.3f}'.format(i_iter, self.args.num_steps, loss_ce_value.avg, loss_fm_value.avg, loss_S_value.avg, loss_D_value.avg, counts.avg))
                loss_ce_value.reset()
                loss_fm_value.reset()
                loss_S_value.reset()
                loss_D_value.reset()
                counts.reset()
            
            if (i_iter % self.args.save_pred_every == 0 and i_iter!=0):
                miou = self.eval(samples='hundred')[0]
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
                    'net_D_state_dict': self.clf_D.module.state_dict(),
                    'idxs_lb': self.idxs_lb,
                    'idxs_unlb': self.idxs_unlb,
                    'rd_idx': rd}, os.path.join(self.args.checkpoint_dir, 'SSL_' + str(rd) + '_' + str(i_iter) + '.pth'))
                       
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
        # elif self.args.dataset == 'brain_slice':
        #     valloader = data.DataLoader(Brain_SliceGTDataSet(self.args.data_dir, self.args.test_data_list, crop_size=(512,384), mean=IMG_MEAN, scale=False, mirror=False),
        #                          batch_size=self.args.test_batch_size, shuffle=False, pin_memory=True)
        #     interp = nn.Upsample(size=(512,384), mode='bilinear', align_corners=True)        

        data_list = []
        from torchvision.utils import save_image
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
                
                # print(output.shape)
                if not self.args.dataset == 'brain_slice':
                    output = output[:,:size[0],:size[1]]    

                # print(output.shape)
                if not self.args.dataset == 'brain_slice':
                    gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int32)
                else:
                    gt = np.asarray(label[0].numpy(), dtype=np.int32)    

                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)

                # print(gt.shape, output.shape, np.unique(gt), np.unique(output))
                data_list.append([gt.flatten(), output.flatten()])
        return get_iou(self.args, data_list)