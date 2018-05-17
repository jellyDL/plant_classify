#!/usr/bin/env python
# encoding: utf-8

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os, sys

from args import *
#from net import *
from net_origin import *
from data_loader import *
from utils import *
from PIL import Image
from test import *

def run_net(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list

    # Create model directory
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4850, 0.4580, 0.4077), (1.0, 1.0, 1.0))
            #transforms.Normalize((123.68, 116.78, 103.94), (1.0, 1.0, 1.0))
        ])
    block_cnt = [int(c) for c in args.blocks.split(",") ]
    net = FancyNet(block_cnt)
    #print "##### net:",net
    # net = ResNet(Bottleneck, [3, 4, 6, 3])

    if torch.cuda.is_available():
        print 'USE CUDA : net'
        net = net.cuda()
    #print net.keys()
    print 'before resume'
    if args.resume_flag == 1 and args.resume_path is not None:
        print 'resume in', args.resume_path
        trained_model = torch.load(args.resume_path)
        '''
        (key, value) = trained_model.popitem()
        print key, 'is remove from trained_model'
        (key, value) = trained_model.popitem()
        print key, 'is remove from trained_model'
        '''
        # print trained_model.keys()
        net.load_state_dict(trained_model)
        print 'resume_successs'
    else:
        print 'no resume path, start from random initialization'

    if args.phase == "test":
        print 'TEST phase'
        net = net.eval()
        test(args, net, transform)
    else:
        print 'TRAIN phase'
        train(args, net, transform)

def train(args, net, transform):
    loss_ratio = args.loss_ratio
    #shuffle = True
    shuffle = False
    data_loader = get_loader(args.root_folder, args.bg_folder,
                             transform = transform, batch_size = args.batch_size,
                             shuffle = shuffle, num_workers = args.num_workers)

    optimizer = torch.optim.Adam(net.parameters(), lr = args.learning_rate, weight_decay = 0.0005)
    step_change_lr_1 = max(1, (args.start_epoch*2 + args.end_epoch*3)//5 - 1)
    step_change_lr_2 = max(1, (args.start_epoch*1 + args.end_epoch*4)//5 - 1)
    step_save_model = max(1, (args.end_epoch - args.start_epoch)//10)
    step_change_loss_ratio = (args.end_epoch - args.start_epoch)//5
    for epoch in range(args.start_epoch, args.end_epoch):
        print "EPOCH", epoch

        for i, (images, score_maps, geo_maps, train_masks) in enumerate(data_loader):
            #print epoch, i
            images = to_cuda(images)
            y_scores = to_cuda(score_maps)
            y_geos = to_cuda(geo_maps)
            y_masks = to_cuda(train_masks)
            time0 = time.time()
            p_scores, p_geos, p_angle = net(images)
            #  time1 = time.time()
            #print "BE LOSS", y_masks.size(), torch.sum(y_masks[0])
            #print "be LOSS", y_scores.size(), torch.sum(y_scores[0])
            seg_loss, dice_loss, balance_loss = EastLoss(p_scores, p_geos, p_angle, y_scores, y_geos, y_masks)
            #  time2 = time.time()
            #  print('net_time :', time1 - time0)
            #  print('loss_time: ', time2 - time1)
            #  print('total_time: ', time2 - time0)
            #  loss = seg_loss + loss_ratio * dice_loss / 2.0 + loss_ratio * balance_loss
            loss = seg_loss + loss_ratio * dice_loss
            #  loss = seg_loss + loss_ratio * dice_loss / 2.0 + loss_ratio * balance_loss
            #  loss = balance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print (get_time_by_sec(False) + 'Epoch [%d/%d], Iter [%d/%d] , Seg: %.6f, Dice: %.6f, Blan: %.6f, Total: %.6f'%(epoch, args.end_epoch, i + 1, data_loader.__len__(), seg_loss.data[0], dice_loss.data[0], balance_loss.data[0], loss.data[0]))

        if epoch == step_change_lr_1:
            print 'change lr to', args.learning_rate * 0.5
            optimizer = torch.optim.Adam(net.parameters(), lr = args.learning_rate * 0.5, weight_decay = 0.0005)
        elif epoch == step_change_lr_2:
            print 'change lr to', args.learning_rate * 0.25
            optimizer = torch.optim.Adam(net.parameters(), lr = args.learning_rate * 0.25, weight_decay = 0.0005)
            #if i == 1:
            #    break
        #break

        #  if (epoch + 1 - args.start_epoch) % step_change_loss_ratio == 0 and loss_ratio > 0.001:
            #  loss_ratio = loss_ratio / 2
        #  if (epoch + 1 - args.start_epoch) % step_change_loss_ratio == 0 and loss_ratio > 0.1:
            #  loss_ratio = loss_ratio - 0.2

        if (epoch + 1 - args.start_epoch) % step_save_model == 0:
             save_path = args.save_folder
             if save_path[-1] == '/':
                save_path = save_path[:-1] + "_"
             else:
                save_path += "_"
             save_path += str(epoch + 1) + ".ckpt"
             save_path = os.path.join(args.save_folder, save_path)
             print 'save_path: ', save_path
             torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    args = load_args()
    print args
    run_net(args)
    '''
    if args.phase == "train":
        print 'TRAIN phase'
        train(args)
    elif args.phase == "test":
        print 'TEST phase'
        test(args)
    else:
        print "not known phase"
    '''
