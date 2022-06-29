from email.policy import strict
import os
from pickle import TRUE
import random
from time import time
import warnings
from datetime import datetime
from collections import OrderedDict
import math
import random
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data 
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from datetime import datetime
import warnings
import data.data_loaders
import data.data_transforms
import data.network_utils
from data.data_loaders import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from utils.image_utils import PSNR
import losses
import matplotlib.pyplot as plt
from basicvsr_nl_deblur import BasicVSRNet_deblur
from warmup_scheduler import GradualWarmupScheduler
import glob
import tqdm
import losses
import cv2
from  torchsummary import summary
from tensorboardX import SummaryWriter


def main():
    torch.backends.cudnn.benchmark  = True
    batch_size=2   
    init_epoch=0
    epoch_num=4000
    data_dir='./dataset_GOPRO'
    now=datetime.now()    
    save_path='./result'
    model=BasicVSRNet_deblur(spynet_pretrained='./spynet_20210409-c6c1bd09.pth')
    criterion1=losses.CharbonnierLoss()
    criterion2=losses.EdgeLoss()
    train_transforms = data.data_transforms.Compose([
        data.data_transforms.Normalize((0.0,),(255.0,)),    
        data.data_transforms.RandomCrop([128,128]), 
        data.data_transforms.RandomVerticalFlip(),
        data.data_transforms.RandomHorizontalFlip(),
        data.data_transforms.RandomColorChannel(),
        data.data_transforms.RandomGaussianNoise([0, 1e-4]),
        data.data_transforms.ToTensor(),])

    test_transforms = data.data_transforms.Compose([
        data.data_transforms.Normalize((0.0,),(255.0,)),
        # data.data_transforms.RandomCrop([128,128]), 
        data.data_transforms.ToTensor(),])
    train_json_path='/Deblurring/data/RealBlur_R_train.json'
    train_dataset = ImageProcessDataset(data.data_loaders.DatasetType.TRAIN,train_json_path,transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_json_path='/Deblurring/data/RealBlur_R_test.json'
    test_dataset = ImageProcessDataset(data.data_loaders.DatasetType.TEST,test_json_path,transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    output_dir = os.path.join(save_path, dt.now().isoformat(), '%s')
    log_dir = output_dir 
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer  = SummaryWriter(os.path.join(log_dir, 'test'))
    Best_Img_PSNR= 0
    init_epoch=0
    epoch_idx=1  
    lr=2e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),eps=1e-8) 
    train(train_loader,test_loader,batch_size, model, criterion1,criterion2, optimizer,init_epoch,epoch_num,lr,train_writer,test_writer,Best_Img_PSNR)
    # test(test_loader,batch_size,epoch_idx,model,test_writer)
def test(test_loader,batch_size,epoch_idx,model,test_writer):
    model.eval()
    init_epoch = epoch_idx+1
    seq_num = len(test_loader)
    batch_time = data.network_utils.AverageMeter()
    test_time = data.network_utils.AverageMeter()
    psnrs_out = data.network_utils.AverageMeter()
    test_time_start=time()
    ckpt_dir='/final_test'
    for seq_idx,sample in enumerate(test_loader):
        seq_blur_cuda=data.network_utils.var_or_cuda(sample['blur'])
        seq_clear_cuda=data.network_utils.var_or_cuda(sample['clear'])
        b,n,c,h,w=seq_blur_cuda.shape   
        with torch.no_grad():
            seq_blur_cuda.cuda()
            seq_clear_cuda.cuda()
            restored=model(seq_blur_cuda)
            restored = torch.clamp(restored,0,1)            
            img_psnr=0
            img_dir=os.path.join(ckpt_dir,sample['name'][0])
            for i  in range (n):
                img_name=os.path.join(img_dir,sample['info'][i][0].split('/')[-1])               
                cv2.imwrite(img_name,(restored[:,i,:,:,:].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8),[int(cv2.IMWRITE_PNG_COMPRESSION), 5])
            img_psnr=PSNR(restored,seq_clear_cuda)
            psnrs_out.update(img_psnr.item(),batch_size)
    print('the test psnr is :',psnrs_out.avg)
    test_writer.add_scalar('/EpochPSNR_0_test',psnrs_out.avg, epoch_idx)
    return psnrs_out.avg

def train(train_loader,test_loader,batch_size, model, criterion1,criterion2, optimizer,init_epoch,epoch_num,lr,train_writer,test_writer,Best_Img_PSNR):        
    optimizer=optimizer  
    model.cuda()
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_num-warmup_epochs, eta_min=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)      
    Best_Img_PSNR_1=0
    for param in model.parameters():
        param.grad = None
    model.train()
    for epoch_idx in range(0,epoch_num):
        print('this is lr:',optimizer.state_dict()['param_groups'][0]['lr'])
        print("this is the %% epoch",epoch_idx)
        epoch_start_time = time()
        losses =data.network_utils.AverageMeter()
        psnrs_out = data.network_utils.AverageMeter()
        for seq_idx,sample in enumerate(train_loader):
            seq_blur_cuda=data.network_utils.var_or_cuda(sample['blur'])
            seq_clear_cuda=data.network_utils.var_or_cuda(sample['clear'])
            seq_blur_cuda.cuda()
            seq_clear_cuda.cuda()               
            ckpt_dir='/result_go_pro/final_test'
            restored=model(seq_blur_cuda)
            b,n,c,h,w=seq_blur_cuda.shape                       
            loss=0.
            img_psnr=0.  
            for i in range (0,n):
                loss=criterion1(restored[:,i,:,:,:],seq_clear_cuda[:,i,:,:,:])+0.05*criterion2(restored[:,i,:,:,:],seq_clear_cuda[:,i,:,:,:])+loss          
            img_psnr=PSNR(restored,seq_clear_cuda)
            psnrs_out.update(img_psnr.item(),batch_size)                    
            losses.update(loss.item(),batch_size)                
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()            
        train_writer.add_scalar('/EpochPSNR_0_TRAIN',psnrs_out.avg, epoch_idx )
        train_writer.add_scalar('/Epochloss_0_TRAIN',losses.avg, epoch_idx )
        if epoch_idx ==0:
            torch.save(model.state_dict(),os.path.join(ckpt_dir, 'best-ckpt.pth.tar'))             

        if epoch_idx % 4==0:        
            test_psnr=test(test_loader,batch_size,epoch_idx,model,test_writer)
            print('this is test psnr:',test_psnr)
            if test_psnr >= Best_Img_PSNR:
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                Best_Img_PSNR = test_psnr
                Best_Img_PSNR_1=Best_Img_PSNR
                Best_Epoch = epoch_idx
                torch.save(model.state_dict(),os.path.join(ckpt_dir, 'best-ckpt.pth.tar')) 
        Best_Img_PSNR=Best_Img_PSNR_1
        if epoch_idx%50==0:
            model_name=str(epoch_idx/50)+'.pth.tar'
            torch.save(model.state_dict(),os.path.join(ckpt_dir, model_name))
        model.train()
        scheduler.step()
        
    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
    main()

