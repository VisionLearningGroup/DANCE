from __future__ import print_function
import yaml
import easydict
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from apex import amp, optimizers
from data_loader.get_loader import get_loader_class_inc
from utils.utils import *
from utils.lr_schedule import inv_lr_scheduler
from utils.loss import *
from models.LinearAverage import LinearAverage
from eval import test, test_class_inc

# Training settings

import argparse

parser = argparse.ArgumentParser(description='Pytorch DA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

parser.add_argument('--source_path', type=str, default='./utils/source_list.txt', metavar='B',
                    help='source data list path')
parser.add_argument('--target_unlabeled_path', type=str, default='./utils/target_list.txt', metavar='B',
                    help='target data list path')
parser.add_argument('--target_labeled_path', type=str, default='./utils/target_list.txt', metavar='B',
                    help='target labeled data list path')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--exp_name', type=str, default='office_close', help='/path/to/config/file')
parser.add_argument('--T', type=int, default=1, metavar='S',
                    help='number of walking (default: 1)')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

# args = parser.parse_args()
args = parser.parse_args()
config_file = args.config
conf = yaml.load(open(config_file))
save_config = yaml.load(open(config_file))
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

args.cuda = torch.cuda.is_available()
source_data = args.source_path
target_data = args.target_unlabeled_path
target_data_labeled = args.target_labeled_path
evaluation_data = args.target_unlabeled_path

batch_size = conf.data.dataloader.batch_size
filename = source_data.split("_")[1] + "2" + target_data.split("_")[1]
filename = os.path.join("record", args.exp_name,
                        config_file.replace(".yaml", ""), filename)
if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
print("record in %s " % filename)

data_transforms = {
    source_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data_labeled: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    evaluation_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = torch.cuda.is_available()
source_loader, target_loader, target_labeled_loader, \
test_loader, target_folder = get_loader_class_inc(source_data, target_data,
                                           target_data_labeled, evaluation_data,
                                           data_transforms, batch_size=batch_size,
                                           return_id=True,
                                           balanced=conf.data.dataloader.class_balance)
dataset_test = test_loader
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_target = conf.data.dataset.n_target
num_class = n_share + n_source_private

G, C1 = get_model_mme(conf.model.base_model, num_class=num_class,
                      temp=conf.model.temp)
_, C2 = get_model_mme(conf.model.base_model, num_class=n_target,
                      temp=conf.model.temp)

device = torch.device("cuda")
G.to(device)
C1.to(device)
C2.to(device)
ndata = target_folder.__len__()

## Memory
lemniscate = LinearAverage(2048, ndata, conf.model.temp, conf.train.momentum).cuda()
params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad and "features" in key:
        if 'bias' in key:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]
        else:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]
    else:
        if 'bias' in key:
            params += [{'params': [value], 'lr': 1.0,
                        'weight_decay': conf.train.weight_decay}]
        else:
            params += [{'params': [value], 'lr': 1.0,
                        'weight_decay': conf.train.weight_decay}]
criterion = torch.nn.CrossEntropyLoss().cuda()

opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                  weight_decay=0.0005, nesterov=True)
opt_c1 = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                   momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                   nesterov=True)
[G, C1, C2], [opt_g, opt_c1] = amp.initialize([G, C1, C2],
                                              [opt_g, opt_c1],
                                              opt_level="O1")
G = nn.DataParallel(G)
C1 = nn.DataParallel(C1)
C2 = nn.DataParallel(C2)
param_lr_g = []
for param_group in opt_g.param_groups:
    param_lr_g.append(param_group["lr"])
param_lr_f = []
for param_group in opt_c1.param_groups:
    param_lr_f.append(param_group["lr"])


def train():
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_l = iter(target_labeled_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_l = len(target_labeled_loader)
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_l == 0:
            data_iter_t_l = iter(target_labeled_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_l = next(data_iter_t_l)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_f, opt_c1, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        index_t = data_t[2]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        index_t = Variable(index_t.cuda())
        img_t_l = data_t_l[0].cuda()
        label_t_l = data_t_l[1].cuda()

        if len(img_t) < batch_size:
            break
        if len(img_s) < batch_size:
            break
        opt_g.zero_grad()
        opt_c1.zero_grad()
        ## Weight normalizztion
        C1.module.weight_norm()
        ## Source loss calculation
        feat = G(img_s)
        out_s = C1(feat)
        loss_s = criterion(out_s, label_s)
        #loss_s += criterion(C2(feat.detach()), label_s)

        feat_t = G(img_t)
        out_t = C1(feat_t)
        feat_t = F.normalize(feat_t)
        ## Train a linear classifier on top of feature extractor.
        ## We should not update feature extractor.
        G.eval()
        feat_t_l = G(img_t_l)
        G.train()
        out_t_l = C2(feat_t_l.detach())
        loss_t_l = criterion(out_t_l, label_t_l)

        ### Calculate mini-batch x memory similarity
        feat_mat = lemniscate(feat_t, index_t)
        ### We do not use memory features present in mini-batch
        feat_mat[:, index_t] = -1 / conf.model.temp
        ### Calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(feat_t,
                                 feat_t.t()) / conf.model.temp
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).bool().cuda()
        feat_mat2.masked_fill_(mask, -1 / conf.model.temp)
        loss_nc = conf.train.eta * entropy(torch.cat([feat_mat,
                                                      feat_mat2], 1))
        loss_ent = conf.train.eta * entropy_margin(out_t, conf.train.thr,
                                                   conf.train.margin)
        all = loss_nc + loss_s + loss_t_l
        with amp.scale_loss(all, [opt_g, opt_c1]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c1.step()
        opt_g.zero_grad()
        opt_c1.zero_grad()
        lemniscate.update_weight(feat_t, index_t)
        if step % conf.train.log_interval == 0:
            print('Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f} '
                  'Loss NC: {:.6f} Loss LT: {:.6f}\t'.format(
                step, conf.train.min_step,
                100 * float(step / conf.train.min_step),
                loss_s.item(), loss_nc.item(), loss_t_l.item()))
        if step > 0 and step % conf.test.test_interval == 0:
            test(step, dataset_test, filename, n_share, num_class, G, C1,
                 conf.train.thr)
            test_class_inc(step, dataset_test, filename, n_target, G, C2,
                           n_share)
            G.train()
            C1.train()
            C2.train()

train()
