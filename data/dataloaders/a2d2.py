# https://github.com/yinwu33/multi_task_learning/blob/ad1185a24b6cd2b58c22a6b3a80951671eef8695/src/datasets/a2d2.py
# https://github.com/ViCE-model/ViCE-model/blob/3a27b51e2419b42c84176df06693de47d6eed9a7/mmsegmentation/tools/convert_datasets/a2d2.py
import os
import cv2
import torch
import json
import random
import numpy as np
import torchvision.transforms as T
import glob
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
#from configs.cfg import args
#from utils.utils import hex_to_rgb
#from datasets.gt_processing import *
from tqdm import tqdm
#from data.augmentation import *


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    red, green, blue = bytes.fromhex(hex)
    return red, green, blue

def get_tuples_recursively(root_path):
    results = [n for n in glob.iglob(root_path+"/camera/**/*.png")] 
    items_list = []
    for camera_image in results:
        label_name = camera_image.replace("_camera_", "_label_").replace("/camera/", "/label/")
        item = (camera_image,#+ '.png'
                label_name)
        items_list.append(item)        
    return items_list

class A2D2(Dataset):
    def __init__(self, data_dir, train_data_list, task='multi_task',  hflip=True, is_crop=True):
        """A2D2 dataset
        # crop_size=(256, 512),
        Args:
            mode (str, optional): 'train', 'val', 'test'. Defaults to 'train'.
            transform (bool, optional): [description]. Defaults to True.
        """
        self.train_list = [line.rstrip('\n') for line in open(train_data_list)]
        self.data_dir = data_dir
        self.task = task
        self.hflip = hflip
        self.is_crop = is_crop
        #self.transform = transform
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4499, 0.4365, 0.4364],
                        std=[0.1902, 0.1972, 0.2085])
        ])
        self.input_size = (512, 256)

        if task == 'val':
            self.val_data = A2D2.load_files(self.data_dir, task='val')
            print('val dataset size: ', len(self.val_data))
        if task == 'train':
            self.train_data = A2D2.load_files(self.data_dir, task='train', train_list=self.train_list)
            print('train dataset size: ', len(self.train_data))

    def __getitem__(self, index):

        if self.task == "val":
            img = cv2.cvtColor(cv2.imread(self.val_data[index]['img_file']), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size, cv2.INTER_LINEAR)
            size = img.shape
            img = self.normalize(img)
            label = cv2.cvtColor(cv2.imread(self.val_data[index]['lbl_file']),
                                 cv2.COLOR_BGR2RGB)
            label = cv2.resize(label, self.input_size, interpolation=cv2.INTER_NEAREST)
            sem_label = A2D2.lbl_convert(label)
            sem_label = torch.from_numpy(sem_label).squeeze().long()
            #print ('img.shape: ', img.shape, 'sem_label.shape: ', sem_label.shape, 'size: ', size)
            return img, sem_label, np.array(size), np.array(size), np.array(size),

        if self.task == 'train':
            img = cv2.cvtColor(cv2.imread(self.train_data[index]['img_file']), cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(cv2.imread(self.train_data[index]['lbl_file']), cv2.COLOR_BGR2RGB)

            h, w, _ = img.shape
            new_size = (int(w/3), int(h/3)) 
            size = img.shape 
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)
            h, w, c = img.shape # h=1208, w=1920

            img = self.normalize(img)
            label = A2D2.lbl_convert(label)
            label = torch.from_numpy(label).squeeze().long()
        
            #print ('before aug: ', img.shape, label.shape)

            if self.is_crop:
                crop_params = T.RandomCrop.get_params(img, output_size=(256, 512))
                a, b, c, d = crop_params
                img = F.crop(img, a, b, c, d)
                label = F.crop(label, a, b, c, d)
            else:
                img = T.resize(img, size=(256,512), interpolation=cv2.INTER_LINEAR)
                label = T.resize(label, size=(256,512), interpolation=cv2.INTER_NEAREST)

            if self.hflip:
                hflip = random.random() < 0.5
                if hflip:
                  F.hflip(img)
                  F.hflip(label)
             
            #size = img.shape
            #print(img.shape, np.array(size))
            #print ('img.shape: ', img.shape, 'label.shape: ', label.shape)

            return img, label, np.array(size), np.array(size), index

    def __len__(self):
        if self.task == 'train':
            return len(self.train_data)
        if self.task == 'val':
            return len(self.val_data)

    @staticmethod
    def load_files(root_dir, task, train_list=[]):
        train_files = []
        val_files = []
        if task == 'train':
            for img_file, lbl_file in A2D2.get_files(root_dir, task, train_list=train_list):
                train_files.append(
                    {
                        'img_file': img_file,
                        'lbl_file': lbl_file
                    }
                )
            return train_files
        if task == 'val':
            items = get_tuples_recursively(os.path.join(root_dir, './val/20180925_112730'))
            for img_path, label_path in items:
                val_files.append(
                    {
                        'img_file': img_path,
                        'lbl_file': label_path
                    }
                )                
            return val_files

    @staticmethod
    def get_files(root_path, task, train_list=None):
        if task == 'train':
            #pool_name = 'pool_' + str(pool) + '_conec_imgs'
            #img_files = [root_path + '/'+pool_name+'/' + i for i in train_list]
            img_files = [root_path + '/train/' + i for i in train_list]
            lbl_files = [ i.replace("_camera_", "_label_").replace("/camera/", "/label/") for i in img_files]
            files = list(zip(img_files, lbl_files))
            return files
        elif task == 'val':
            dir_path = os.path.join(root_path, './val/20180925_112730') # validations split fixed
            all_f = open(os.path.join(dir_path, 'val.txt'), 'r')
            img_files = []
            lbl_files = []
            for f in all_f:
                if 'camera' in f:
                    img_files.append(os.path.join(dir_path, f[2:-1]))
                if 'label' in f:
                    lbl_files.append(os.path.join(dir_path, f[2:-1]))
            img_files.sort()
            lbl_files.sort()
            files = list(zip(img_files, lbl_files))
            return files

    @staticmethod
    def lbl_convert(label):        
        a2d2_color_seg = {
            "#ff0000": 1, "#c80000": 1, "#960000": 1, "#800000": 1, "#b65906": 2, "#963204": 2,
            "#5a1e01": 2, "#5a1e1e": 2, "#cc99ff": 3, "#bd499b": 3, "#ef59bf": 3, "#ff8000": 4,
            "#c88000": 4, "#968000": 4, "#00ff00": 11, "#00c800": 11, "#009600": 11, "#0080ff": 5,
            "#1e1c9e": 5, "#3c1c64": 5, "#00ffff": 6, "#1edcdc": 6, "#3c9dc7": 6, "#ffff00": 13,
            "#ffffc8": 13, "#e96400": 14, "#6e6e00": 0, "#808000": 17, "#ffc125": 15, "#400040": 18,
            "#b97a57": 12, "#000064": 18, "#8b636c": 7, "#d23273": 0, "#ff0080": 18, "#fff68f": 14,
            "#960096": 0, "#ccff99": 18, "#eea2ad": 12, "#212cb1": 14, "#b432b4": 0, "#ff46b9": 18,
            "#eee9bf": 0, "#93fdc2": 8, "#9696c8": 18, "#b496c8": 7, "#48d1cc": 18, "#c87dd2": 15,
            "#9f79ee": 16, "#8000ff": 15, "#ff00ff": 0, "#87ceff": 9, "#f1e6ff": 10, "#60458f": 18,
            "#352e52": 18
        }

        label = np.array(label)
        label_convert = np.zeros_like(label, dtype=np.uint8)

        for key, value in a2d2_color_seg.items():
            index = np.where(np.all(label == hex_to_rgb(key), axis=-1))
            label_convert[index] = value

        #print(label_convert.shape) 
 
        return cv2.cvtColor(label_convert, cv2.COLOR_RGB2GRAY)

if __name__ == '__main__':
    lbl_to_bbox_json('./datasets/a2d2/camera_lidar_semantic')
