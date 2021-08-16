from __future__ import print_function

import argparse
import os.path as osp
import sys
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import embed_net
from utils.data_utils import *
from utils.eval_utils import *
from utils.misc import Logger, set_seed

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Testing')
### dataloader config
parser.add_argument('--dataset', default='sysu', help='dataset name: [regdb or sysu]')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--img_h', default=288, type=int, help='img height')
parser.add_argument('--img_w', default=144, type=int, help='img width')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, help='testing batch size')
### directory config
parser.add_argument('--save_path', default='log/', type=str, help='parent save directory')
parser.add_argument('--exp_name', default='exp', type=str, help='child save directory')
parser.add_argument('--model_name', default='ep_80', type=str,help='model save path')
### model/training config
parser.add_argument('--method', default='full', type=str, help='method type: [baseline or full]')
### evaluation protocols
parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
### misc
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--nvidia_device', default=0, type=int, help='gpu device to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.nvidia_device)

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/workspace/dataset/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = '/workspace/dataset/RegDB/'
    n_class = 206
    test_mode = [2, 1]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
pool_dim = 2048

print('==> Building model..')
net = embed_net(args, n_class)
net.to(device)
cudnn.benchmark = True

sys.stdout = Logger(osp.join(args.save_path, '{}/os_test.txt'.format(args.exp_name)))
    
print('==> Loading data..')
# Data loading code
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

end = time.time()

def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, data in enumerate(gall_loader):
            input = data['img']
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool = net(input, input, test_mode[0], use_cmalign=False)['feat4_p_norm']
            gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_pool
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, data in enumerate(query_loader):
            input = data['img']
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool = net(input, input, test_mode[1], use_cmalign=False)['feat4_p_norm']
            query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool

if dataset == 'sysu':
    print('==> Resuming from checkpoint..')
    
    model_path = osp.join(args.save_path, args.exp_name)
    model_path = osp.join(model_path, 'checkpoints/{}.t'.format(args.model_name))
    
    if os.path.isfile(model_path):
        print('==> loading checkpoint {} from {}'.format(args.model_name, args.exp_name))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint')
    else:
        print('==> checkpoint {} is not found'.format(args.model_name))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_pool = extract_gall_feat(trial_gall_loader)

        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

cmc_pool = all_cmc_pool/10
mAP_pool = all_mAP_pool/10
mINP_pool = all_mINP_pool/10
print('All Average:')
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
