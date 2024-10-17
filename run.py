# Active Learning for Semantic Segmentation (ASL)
import os
import numpy as np
import cv2
import pickle
import argparse
import yaml

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model.deeplabv3p import DeepV3PlusW38
from model.discriminator import S4GAN_D
from query_strategies import Strategy, RandomImage, EntropyImage, CoreSet, EquAL, RandomImageSSL, EntropyImageSSL, CoreSetSSL, EquALSSL

SEED = 0 
RESTORE_FROM = './pretrained_models/wider_resnet38.pth.tar'
#RESTORE_FROM = './pretrained_models/resnet101-5d3b4d8f.pth'

EXP_NAME = 'ALSS'

parser = argparse.ArgumentParser(description='Active Learning for Semantic Segmentation')
parser.add_argument("--resume-path", type=str, default=None, help="resume checkpoint path")
parser.add_argument("--resume-rd-idx", type=int, default=0, help="resume round index")
parser.add_argument("--config", type=str, default='./configs/datasets/a2d2.yaml', help="config file")
parser.add_argument("--pool", type=int, default=5, help="pool index, currently working for A2D2")
parser.add_argument("--split", type=int, default=1, help="split index, currently working for A2D2")
parser.add_argument('--num-classes', default=21, type=int, help='Number of classes in the dataset')

# Budget parameters
parser.add_argument('--initial-image-budget', default=0.1, type=float, help='First training image budget')
parser.add_argument('--sampling-image-budget', default=0.1, type=float, help='Sampling image budget')
parser.add_argument('--num-rounds', default=4, type=int, help='Number of sampling rounds')
parser.add_argument('--lab-percent', default=100, type=int, help='Percentage of dataset considered for the experiment')


# general training parameters
parser.add_argument('--num-epochs', default=1, type=int, help='number of epochs') # use 100 epochs 
parser.add_argument('--train-batch-size', default=16, type=int, help='training batch size')
parser.add_argument('--test-batch-size', default=1, type=int, help='testing batch size')
parser.add_argument("--save-pred-every", type=int, default=1000, help="Save summaries and checkpoint every often.")

parser.add_argument("--checkpoint-dir", type=str, default=None, help="Where to save checkpoints of the model.")
parser.add_argument("--save-dir", type=str, default=None, help="Where to save results and tensorboards")
parser.add_argument("--exp-name", type=str, default=EXP_NAME, help="Name of the experiment")

parser.add_argument('--num-steps', default=15000, type=int, help='number of iterations')
parser.add_argument('--lab-only', action='store_true', help='if using only labeled samples')

parser.add_argument('--verbose', action='store_true', help='show progress bar')
parser.add_argument('--seed', default=SEED, type=int, help='seed index')

# Methods (Image-based)
parser.add_argument('--random-image', action='store_true', help='Random Image sampling')
parser.add_argument('--random-image-SSL', action='store_true', help='Random Image SSL sampling')
parser.add_argument('--entropy-image', action='store_true', help='Entropy Image sampling')
parser.add_argument('--entropy-image-SSL', action='store_true', help='Entropy Image sampling')
parser.add_argument('--coreset', action='store_true', help='Coreset sampling')
parser.add_argument('--coreset-SSL', action='store_true', help='Coreset sampling')
parser.add_argument('--equal', action='store_true', help='EquAL sampling')
parser.add_argument('--equal-SSL', action='store_true', help='EquAL sampling')
#parser.add_argument('--bald-image', action='store_true', help='BALD Image sampling')
#parser.add_argument('--bald-image-SSL', action='store_true', help='BALD Image sampling')
parser.add_argument('--final-dropout', action='store_true', help='enable the dropout layersin the final layers')

parser.add_argument('--num-cycles', default=5, type=int, help='number of sgdr cycles for ensembles')
parser.add_argument('--train-iterations', type=int, default=10000, help='Number of training iterations')

# Data 
parser.add_argument("--train-data-list", type=str, default=None, help="Path to the file listing the images in the dataset.")
parser.add_argument("--data-dir", type=str, default=None, help="Path to the directory containing the PASCAL VOC dataset.")

parser.add_argument("--input-size", type=str, default='321, 321', help="Comma-separated string with height and width of images.") 
parser.add_argument("--ignore-label", type=float, default=255, help="label value to ignored for loss calculation")
parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

# Model
parser.add_argument("--learning-rate-D", type=float, default=1e-4, help="Base learning rate for discriminator.")
parser.add_argument('--learning-rate', default=1e-3, type=float, help='learning rate')
parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.") 
parser.add_argument("--weight-decay", type=float, default=5e-4, help="Regularisation parameter for L2-loss.")

parser.add_argument("--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training.")
parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")

#SSL hyperparameters
parser.add_argument("--lambda-fm", type=float, default=0.1, help="lambda_fm for feature-matching loss.")
parser.add_argument("--lambda-st", type=float, default=1.0, help="lambda_st for self-training.")
parser.add_argument("--threshold-st", type=float, default=0.6, help="threshold_st for the self-training threshold.")

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
args.pool_dir = config['pool_dir']
args.data_dir = config['dataset']['dataset_dir']
args.dataset = config['dataset']['name']
args.num_classes = config['dataset']['num_classes']
args.ignore_label = config['dataset']['ignore_label']
args.restore_from = config['model']['restore_from']
args.input_size = config['dataset']['input_size']

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = True

if args.dataset == 'a2d2':
    args.train_data_list = './data/pools/a2d2/pool_' + str(args.pool) + 'f/split_1/lab_images_100.txt'
elif args.dataset == 'pascal_voc': 
    args.train_data_list = './data/pools/pascal_voc/train_aug.txt'
    args.test_data_list = './data/pools/pascal_voc/val.txt'
# load dataset
print('number of initial images: {}'.format(args.initial_image_budget))
list_train_imgs_orig = [line.rstrip('\n') for line in open(args.train_data_list)] 

args.checkpoint_dir = './log/checkpoints/' + args.dataset + '_' + args.exp_name
args.save_dir = './log/results/' + args.dataset + '_' + args.exp_name
#args.tensorboard_dir = './log/tensorboards/' + args.dataset + '_' + args.exp_name

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

writer = SummaryWriter(log_dir=args.save_dir)

num_train_imgs = len(list_train_imgs_orig)
print ('Training size set: ', num_train_imgs)

idxs_tmp = np.arange(num_train_imgs)
np.random.shuffle(idxs_tmp)

list_train_imgs = [list_train_imgs_orig[item] for item in idxs_tmp] # shuffle list train images
num_labeled = int(len(list_train_imgs)*args.initial_image_budget)

#net = Res_Deeplab()
# if args.bald_image or args.bald_image_SSL:
#     from model.deeplabv3p_mc import DeepV3PlusW38
net =  DeepV3PlusW38(num_classes=args.num_classes) 
net_D = S4GAN_D(num_classes=args.num_classes, dataset=args.dataset)

if args.random_image:
    strategy = RandomImage(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, args, writer)
if args.random_image_SSL:
    strategy = RandomImageSSL(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, net_D, args, writer)
if args.entropy_image_SSL:
    strategy = EntropyImageSSL(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, net_D, args, writer)
if args.entropy_image:
    strategy = EntropyImage(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, args, writer)
if args.coreset:
    strategy = CoreSet(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, args, writer)
if args.coreset_SSL:
    strategy = CoreSetSSL(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, net_D, args, writer)
if args.equal:
    strategy = EquAL(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, args, writer)
if args.equal_SSL:
    strategy = EquALSSL(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, net_D, args, writer)
#if args.bald_image:
#    args.final_dropout = True
#    strategy = BALDImage(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, args, writer)
#if args.bald_image_SSL:
#    args.final_dropout = True
#    strategy = BALDImageSSL(list_train_imgs, idxs_tmp[:num_labeled], idxs_tmp[num_labeled:], net, net_D, args, writer)
## include BALD and BatchBALD

#resume_rd_idx = 1

if args.resume_path:
    print ('resuming checkpoint...')
    checkpoint = torch.load(args.resume_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    resume_rd_idx = checkpoint['rd_idx'] + 1
    strategy.idxs_lb = checkpoint['idxs_lb']
    if 'SSL' in args.resume_path:
        #net_D.load_state_dict(checkpoint['net_D_state_dict'])
        strategy.idxs_unlb = checkpoint['idxs_unlb']

# print info
print('Dataset: ', args.dataset)
print('SEED {}'.format(args.seed))

strategy.args.num_steps = int((args.num_epochs*num_labeled)/args.train_batch_size)
print ('Num Steps: ', strategy.args.num_steps)
miou_list = np.zeros(args.num_rounds+1)

for rd_idx in range(args.resume_rd_idx, args.num_rounds+1):
    print('Round {}'.format(rd_idx))

    if rd_idx>0: 
        strategy.query_and_update()
    print ('num labeled samples: ', len(strategy.idxs_lb), ' num unlabeled samples: ', len(strategy.idxs_unlb))
    
    strategy.args.num_steps = int((args.num_epochs*strategy.idxs_lb.shape[0])/args.train_batch_size)
    print ('Num Steps: ', strategy.args.num_steps)
    
    miou_list[rd_idx] = strategy.train(rd=rd_idx)
    writer.add_scalar('val_miou', miou_list[rd_idx], rd_idx)

print('mIOUs: ', miou_list)
