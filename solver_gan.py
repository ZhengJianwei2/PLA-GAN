
# coded by jiawei jiang
# All rights reserved.
''' This code is the implementation of Low-Dose CT Reconstruction Via Optimization-Inspired GAN'''

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from prep import printProgressBar
import random

from model.PLAGAN.PLA_GAN import Net
from model.PLAGAN.PLA_GAN import discriminator 

from measure import compute_measure
from loss import CharbonnierLoss, PerceptualLoss, EdgeLoss
import lpips


######### Set Seeds ###########
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_arr(arr,patch_size,stride=32):    ## 512*512 to 64*64
    pad = (16, 16, 16, 16) # pad by (0, 1), (2, 1), and (3, 3)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _,_,h,w = arr.shape
    num = h//stride - 1
    arrs = torch.zeros(num*num,1,patch_size,patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):  ## from 64*64 to size 512*512
    arr = torch.zeros(size, size)
    n,_,h,w = arrs.shape
    num = size//stride
    for i in range(num):
        for j in range(num):
            arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,16:48,16:48]
  #return arr
    return arr.unsqueeze(0).unsqueeze(1)

class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size
        self.bolcknum = args.blocknum
        self.batch_size = args.batch_size
        self.patch_n = args.patch_n
        self.batch =  self.batch_size * self.patch_n
  ### model ###
        self.Net = Net(imgsize=self.patch_size, batch = self.batch, blocknum=self.bolcknum,in_channel=1, n_feat=48).cuda()
        self.model_D = discriminator().cuda()

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            self.Net = nn.DataParallel(self.Net)   ## data parallel  ,device_ids=[2,3]

        self.Net.to(self.device)
    
        self.lr = args.lr

        self.lossc = CharbonnierLoss().cuda()
        self.lossm = nn.MSELoss()
        self.losse = EdgeLoss().cuda()
        self.lossp = PerceptualLoss().cuda()
        self.optimizer_G = optim.Adam(self.Net.parameters(), self.lr, betas=(0.9, 0.999),eps=1e-8)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), self.lr, betas=(0.9, 0.999),eps=1e-8)
        self.lpips = lpips.LPIPS(net='alex')

    def save_model(self, iter_):
        # f = os.path.join(self.save_path, 'T2T_vit_{}iter.ckpt'.format(iter_))
        f = os.path.join(self.save_path, 'JJW_{}iter.ckpt'.format(iter_))
        torch.save(self.Net.state_dict(), f)


    def load_model(self, iter_):
        device = torch.device('cpu')
        f = os.path.join(self.save_path, 'JJW_{}iter.ckpt'.format(iter_))
        self.Net.load_state_dict(torch.load(f, map_location=device),strict=False)



    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result,lpips_ori,lpips_pred):
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f} \nLPIPS: {:.6f}".format(original_result[0],
                                                                            original_result[1],
                                                                            original_result[2], lpips_ori*10,fontsize=20))
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f} \nLPIPS: {:.6f}".format(pred_result[0],
                                                                            pred_result[1],
                                                                            pred_result[2], lpips_pred*10,fontsize=20))
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    
    
    def train(self):

        train_losses = []
        total_iters = 0
        start_time = time.time()
        loss_all = []
        real_labels_patch = Variable(torch.ones( self.patch_n*  self.batch_size, 36) -0.05).cuda()
        fake_labels_patch = Variable(torch.zeros(self.patch_n*  self.batch_size, 36)).cuda()
        for epoch in range(1, self.num_epochs):
            self.Net.train(True)
            D_loss_sum_patch = 0
            epoch_loss = 0

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)   ## expand one dimension given the dimension 0  4->[1,4]
                y = y.unsqueeze(0).float().to(self.device)   ## copy data to device

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)  ## similar to reshape
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.Net(x)
                real_output = self.model_D(y)
                self.D_loss_real = self.lossm(real_output, real_labels_patch)

                fake_output = self.model_D(pred)
                self.D_loss_fake =  self.lossm(fake_output, fake_labels_patch)
                D_loss_patch = (self.D_loss_real + self.D_loss_fake)/2
                D_loss_sum_patch += D_loss_patch.item()
                self.optimizer_D.zero_grad()  
                D_loss_patch.backward()
                self.optimizer_D.step()
            
        ########################################################################
                
                pred = self.Net(x)
                fake_output = self.model_D(pred)

                G_loss_patch = self.lossm(fake_output, real_labels_patch)
                loss1 = self.lossc(pred, y)
                loss2 = self.lossp(pred, y)
                loss3 = self.losse(pred, y)
                loss = 1*loss1 + 0.06*loss2 + 0.05*loss3 + 0.2*G_loss_patch

                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()
                epoch_loss += loss.item()

                train_losses.append(loss.item())

                loss_all.append(loss.item())
                sys.stdout.write(
                    "\r[STEP [%d], EPOCH [%d/%d], EPOCH_loss:%f, Loss_G: %f, Loss_D: %f ITER [%d/%d], TIME: %fs] "
                    %  (
                        total_iters,
                        epoch,self.num_epochs,                       
                        epoch_loss,
                        loss,
                        D_loss_patch,
                        iter_+1,len(self.data_loader),
                        time.time() - start_time,
                        )
                )       
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model

                if total_iters % 2000 == 0:
                    print("save model: ",total_iters)
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
            
        self.save_model(total_iters)
        np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
        print("total_iters:",total_iters)
        ## save loss figure
        plt.plot(np.array(loss_all), 'r')  ## print out the loss curve
        plt.savefig('save/loss.png')

    def test(self):
        del self.Net
        self.Net = Net(imgsize=self.patch_size ,blocknum=self.bolcknum,in_channel=1, n_feat=48).cuda()
        self.Net.eval()

        # self.SUNet.eval()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            self.Net = nn.DataParallel(self.Net)
        self.Net.to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg , ori_lpips_avg = 0, 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg , pred_lpips_avg= 0, 0, 0, 0 

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                
                arrs = split_arr(x, 64).to(self.device)  ## split to image patches for test into 4 patches

                arrs[0:64] = self.Net(arrs[0:64])
                arrs[64:2*64] = self.Net(arrs[64:2*64])
                arrs[2*64:3*64] = self.Net(arrs[2*64:3*64])
                arrs[3*64:4*64] = self.Net(arrs[3*64:4*64])

                pred = agg_arr(arrs, 512)
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min
                lpips_ori = self.lpips.forward(x,y).item()
                lpips_pred = self.lpips.forward(pred,y).item()

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                ori_lpips_avg += lpips_ori 
                
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]
                pred_lpips_avg  += lpips_pred

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result,lpips_ori,lpips_pred)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.6f} \nLPIPS avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader),ori_lpips_avg*10/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.6f} \nLPIPS avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                  pred_ssim_avg/len(self.data_loader), 
                                                                                                  pred_rmse_avg/len(self.data_loader),pred_lpips_avg*10/len(self.data_loader) ))
