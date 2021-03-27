import argparse
import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import tqdm
import json
import utils
#import sklearn.preprocessing
import numpy as np
import random
import os
import copy
import math
import matplotlib.pyplot as plt
from u2net import u2net
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed
# python -m torch.distributed.launch --nproc_per_node=4 train.py
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
from nnAudio import Spectrogram
from denseunet import denseunet
from hardnet import hardnet
import warnings
warnings.filterwarnings('ignore')


tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True

def diceloss(input,target):

    input = F.sigmoid(input)
    intersection = input*target
    dice_loss = (2.*intersection.sum()/(input.sum()+target.sum()+1)).mean()
    return 1-dice_loss

def compute_amp(x):
    x = x.reshape(x.shape[0], -1, 2, x.shape[-2], x.shape[-1])
    x = torch.norm(x, dim=2)
    return x

def myloss(input,target):

    loss_phase = F.mse_loss(input, target)
    # loss_amp = F.mse_loss(compute_amp(input),compute_amp(target))
    # loss = loss_phase + loss_amp

    return loss_phase

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train(args, model, device, train_loader, optimizer, scaler):
    losses = []
    losses1 = []
    losses2 = []
    model.train()

    i = 0
    for x, y in tqdm.tqdm(train_loader,disable=not(args.local_rank==0)):
        i = i+1
        x, y = x.to(device), y.to(device)

        X = utils.Spectrogram(utils.STFT(x, device, args.fft, args.hop))
        Y = utils.Spectrogram(utils.STFT(y, device, args.fft, args.hop))
        X = X[:,:,:args.bins,:]
        Y = Y[:,:,:args.bins,:]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            Y_est = model(X)
            loss = myloss(Y_est,  Y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        reduced_loss = reduce_mean(loss.data,args.nprocs)
        losses1.append(reduced_loss.item())
        losses2.append(reduced_loss.item())
        losses.append(reduced_loss.item())

        if i%5 == 0:
            X = utils.Spectrogram(utils.STFT(x-y, device, args.fft, args.hop))
            X = X[:,:,:args.bins,:]
            Y = torch.zeros_like(X)
            # Y = X

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                Y_est = model(X)
                loss = myloss(Y_est,  Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # reduced_loss = reduce_mean(loss.data,args.nprocs)

        # reduced_loss1 = reduce_mean(loss1.data,args.nprocs)
        # losses1.append(reduced_loss1.item())
        # reduced_loss2 = reduce_mean(loss2.data,args.nprocs)
        # losses2.append(reduced_loss2.item())

        # break

    return np.array(losses).mean(),np.array(losses1).mean(),np.array(losses2).mean()


def get_parser():
    parser = argparse.ArgumentParser(description='Music Separation Training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--nprocs', type=int)
    parser.add_argument('--root', type=str, default='/data/nx/musdb16')
    # parser.add_argument('--root', type=str, default='../dataset/musdb44')
    parser.add_argument('--model', type=str, default="./models/aa")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--target', type=str, default='vocals') #accompaniment vocals

    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--samples-per-track', type=int, default=100)

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=210)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--seed', type=int, default=42, metavar='S')

    # Model Parameters
    parser.add_argument('--dur', type=int, default=256)

    parser.add_argument('--fft', type=int, default=1024)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--bins', type=int, default=None)

    parser.add_argument('--nb-workers', type=int, default=12)
    parser.add_argument('--device', type=str, default="cuda")

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()
    
    if args.nprocs is None:
        args.nprocs = torch.cuda.device_count()
    if args.bins is None:
        args.bins = args.fft//2 + 1

    model_path = Path(args.model)
    model_path.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    #main_worker(args.local_rank, args.nprocs, args)
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

def main_worker(local_rank,nprocs,args):
    # local_rank = local_rank + 1
    args.local_rank = local_rank
    device = args.device
    torch.cuda.set_device(local_rank)
    #distributed.init_process_group(backend='nccl')
    distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=args.nprocs, rank=local_rank)

    train_dataset = data.load_datasets(args)
    train_sampler=DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
        num_workers=args.nb_workers,pin_memory=True, sampler=train_sampler
    )

    # model = u2net(2,2,args.bins).to(device)
    # model = mmu2net(2,2,args.bins).to(device)
    model = u2net(2,2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
    scaler = torch.cuda.amp.GradScaler()
    es = utils.EarlyStopping(patience=args.patience)

    if args.pretrained:
        with open(Path(args.model, args.target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(args.model, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        # model.load_state_dict(torch.load(Path(args.model, args.target + ".pth"), map_location=device))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
 
        epochs = range(results['epochs_trained']+1, args.epochs + 1)
        train_losses = results['train_loss_history']
        losses1 = []
        losses2 = []
        best_epoch = results['best_epoch']
        es.best = results['best_loss']
        es.num_bad_epochs = results['num_bad_epochs']
    else:
        epochs = range(1, args.epochs + 1)
        train_losses = []
        losses1 = []
        losses2 = []

        best_epoch = 0

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    pbar = tqdm.tqdm(epochs)
    for epoch in pbar:

        train_sampler.set_epoch(epoch)
        train_loss, loss1, loss2 = train(args, model, device, train_loader, optimizer, scaler)
        scheduler.step()

        train_losses.append(train_loss)
        losses1.append(loss1)
        losses2.append(loss2)

        # pbar.set_postfix(loss=train_loss,lr=optimizer.state_dict()['param_groups'][0]['lr'])

        stop = es.step(train_loss)
        best_epoch = epoch if train_loss == es.best else best_epoch

        pbar.set_postfix(loss=train_loss,l2=loss2,l1=loss1)

        if args.local_rank == 0:
            utils.save_checkpoint({
                'epoch': epoch ,
                'state_dict': model.module.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            },
                is_best = train_loss == es.best,
                path=Path(args.model),
                target=args.target
            )

            # save params
            params = {
                'epochs_trained': epoch,
                'args': vars(args),
                'best_loss': es.best,
                'best_epoch': best_epoch,
                'train_loss_history': train_losses,
                'loss1_history': losses1,
                'loss2_history': losses2,
                'num_bad_epochs': es.num_bad_epochs
            }

            with open(Path(args.model, args.target + '.json'), 'w') as outfile:
                outfile.write(json.dumps(params, indent=4, sort_keys=True))

        if stop:
            print("Apply Early Stopping")

            break


if __name__ == "__main__":
    main()
