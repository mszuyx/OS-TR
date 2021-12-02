import numpy as np
import logging
import sys
import time
import argparse
import os

import torch
from torch import nn
from torchsummary import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.data import datasets
from utils.model import models
from utils.evaluate import Evaluator
from utils.loss import myloss
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter   

def plot_images(writer, inputs, patch, target, scores, smooth_pred, epoch):
    writer: SummaryWriter
    writer.add_images('seg/inputs', torch.concat(inputs, 0), global_step=epoch)
    writer.add_images('seg/patches', torch.concat(patch, 0), global_step=epoch)
    writer.add_images('seg/target', torch.unsqueeze(torch.concat(target, 0), 1), global_step=epoch)
    writer.add_images('seg/scores', torch.unsqueeze(torch.concat(scores, 0), 1), global_step=epoch)
    writer.add_images('seg/prod_scores', torch.unsqueeze(torch.concat(smooth_pred, 0), 1), global_step=epoch)

def main(seed=2018, epoches=100): #80
    parser = argparse.ArgumentParser(description='my_trans')
    # dataset option
    parser.add_argument('--dataset_name', type=str, default='tcd_alot_dtd', choices=['dtd','tcd','tcd_alot_dtd'], help='dataset name')
    parser.add_argument('--model_name', type=str, default='OSnet_mb_frozen', choices=['ResNet50_frozen', 'ResNet50_free'], help='model name')
    parser.add_argument('--loss_name', type=str, default='weighted_bce', choices=['weighted_bce', 'DF'], help='set the loss function')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--checkname', type=int, default=0, help='set the checkpoint name (default: 0)')
    parser.add_argument('--train_batch_size', type=int, default=32, metavar='N', help='input batch size for training (default: 24)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='N', help='input batch size for testing (default: 4)')
    parser.add_argument('--load_pre_train', type=str, default=None, 
                        help='load from pth file')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Data augmentation settings
    transform_train_ = A.Compose([
        # A.Normalize (mean=(0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667),p=1),
        A.HorizontalFlip(p=0.5),
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.8,1.2), translate_percent=0.1, rotate=(-20,20), shear=(-20,20), p=0.3),
        A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.5),
        A.RGBShift(r_shift_limit=(-80,80), g_shift_limit=(-80,80), b_shift_limit=(-80,80), p=0.5),
        A.transforms.Emboss (alpha=(0.5, 1.0), strength=(0.7, 1.0), p=0.1),
        ToTensorV2()
    ])

    transform_ref_ = A.Compose([
        # A.Normalize (mean=(0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667),p=1),
        A.HorizontalFlip(p=0.5),
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.5),
        A.RGBShift(r_shift_limit=(-80,80), g_shift_limit=(-80,80), b_shift_limit=(-80,80), p=0.5),
        A.transforms.Emboss (alpha=(0.5, 1.0), strength=(0.7, 1.0), p=0.3),
        ToTensorV2()
    ])

    transform_valid_ = A.Compose([
        # A.Normalize (mean=(0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667),p=1),
        ToTensorV2()
    ])

    # Setup data generator
    evaluator = Evaluator(num_class=6) # why not 5?
    mydataset_embedding = datasets[args.dataset_name]
    data_val = mydataset_embedding(split='test', transform = transform_valid_, transform_ref = transform_valid_, checkpoint=args.checkname)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=args.test_batch_size, num_workers = 16, pin_memory=True, shuffle=False)
    data_train = mydataset_embedding(split='train', transform = transform_train_, transform_ref = transform_ref_, checkpoint=args.checkname)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.train_batch_size, num_workers = 16, pin_memory=True, shuffle=True)

    dir_name = 'log/' + str(args.dataset_name) + '_' + str(args.model_name) + '_' + str(args.loss_name) + '_' + 'LR_' + str(args.lr)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logging.basicConfig(level=logging.INFO,
                        filename=dir_name + '/output_' + now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('dataset_name: %s, model_name: %s, loss_name: %s, batch_size: %s', args.dataset_name, args.model_name, args.loss_name, args.train_batch_size)
    logging.info('test with: %s', data_val.test)

    writer = SummaryWriter('/home/ros/OS_TR/runs/' + str(args.dataset_name) + '_' + str(args.model_name) + '_' + str(args.loss_name) + '_' + 'LR_' + str(args.lr) +"_"+ str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))

    # Complie model
    if args.load_pre_train is not None:
        print('Loading weights from: ' + args.load_pre_train)
        model = torch.load(args.load_pre_train)
        print('============== Model loaded ==============')
    else:
        model = models[args.model_name]()
    
    # CUDA init
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
    # summary(model, [(3,256,256),(3,256,256)])
    model.train()

    # Setup loss function & optimizer, scheduler
    criterion = myloss[args.loss_name]()
    optim_para = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.SGD(optim_para, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(optim_para,lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8) #10 , 0.8
    # plat_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=20, factor=.5)

    # Init loss & IoU
    IoU_final = 0
    epoch_final = 0
    losses = 0
    iteration = 0

    # Start training
    for epoch in range(epoches):
        # scheduler.step()
        train_loss = 0
        logging.info('epoch:' + str(epoch))
        start = time.time()
        np.random.seed(epoch)
        for i, data in enumerate(loader_train):
            _, _, inputs, target, patch, _ = data[0], data[1], data[2], data[3], data[4], data[5]
            # inputs = inputs.float()
            # patch = patch.float()
            iteration += 1
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
                patch = patch.cuda()

            output = model(inputs, patch)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses += loss.item()

            if iteration % 20 == 0:
                run_time = time.time() - start
                start = time.time()
                losses = losses / 20
                logging.info('iter:' + str(iteration) + " time:" + str(run_time) + " train loss = {:02.5f}".format(losses))
                writer.add_scalar("errors/batch_error", losses, iteration)
                losses = 0
        snapshot_path = dir_name + '/snapshot-epoch_{epoches}_texture.pth'.format(epoches=now_time)
        writer.add_scalar("errors/train_loss", train_loss, epoch)
        # print(iteration)
        # Model evaluation after one epoch
        model.eval()
        with torch.no_grad():
            evaluator.reset()
            np.random.seed(2019)

            inps, tars, pats,  preds,preds_smooth = [],[],[],[],[]

            for i, data in enumerate(loader_val):
                _, _, inputs, target, patch, image_class = data[0], data[1], data[2], data[3], data[4], data[5]
                # inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda(non_blocking=True)
                    patch = patch.cuda()

                scores = model(inputs, patch)


                if i < 4/args.test_batch_size:
                    inps.append(inputs)
                    pats.append(patch)
                    tars.append(target)
                    preds.append(scores[:, 0, :, :])
                    preds_smooth.append(torch.clone(scores[:, 0, :, :].detach()))

                scores[scores >= 0.5] = 1
                scores[scores < 0.5] = 0
                seg = scores[:, 0, :, :].long()
                pred = seg.data.cpu().numpy()
                target = target.cpu().numpy()

                # Add batch sample into evaluator
                evaluator.add_batch(target, pred, image_class)

            mIoU, mIoU_d = evaluator.Mean_Intersection_over_Union()
            FBIoU = evaluator.FBIoU()

            plot_images(writer, inps, pats, tars, preds, preds_smooth, epoch)

            writer.add_scalar("errors/mIoU", mIoU, epoch)
            writer.add_scalar("errors/FBIoU", FBIoU, epoch)
            writer.add_scalars("errors/mIoU_d", dict(zip(data_val.test, mIoU_d)), epoch)

            logging.info("{:10s} {:.3f}".format('IoU_mean', mIoU))
            logging.info("{:10s} {}".format('IoU_mean_detail', mIoU_d))
            logging.info("{:10s} {:.3f}".format('FBIoU', FBIoU))
            if mIoU > IoU_final:
                epoch_final = epoch
                IoU_final = mIoU
                # torch.save(model.state_dict(), snapshot_path)
                torch.save(model, snapshot_path)
            logging.info('best_epoch:' + str(epoch_final))
            logging.info("{:10s} {:.3f}".format('best_IoU', IoU_final))
        
        model.train()
        scheduler.step()
        # plat_scheduler.step(mIoU)
        logging.info(f"LR: {optimizer.param_groups[0]['lr']}")

    logging.info(epoch_final)
    logging.info(IoU_final)
    final_path = dir_name + '/final_{epoches}_texture.pth'.format(epoches=now_time)
    torch.save(model, final_path)


if __name__ == '__main__':
    main()