from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import torchsummary
import datetime
import tqdm
import cv2
import math
import albumentations as albu
from pathlib import Path
from tensorboardX import SummaryWriter
import MyDataset as MyD
from models import basic, stackhourglass, submodule
from processing import display
import transforms

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=96,
                    help='maxium disparity')
parser.add_argument('-use-hourglass',type = bool,default=True,
                    help='enable hourglass')
parser.add_argument('--datapath', default='/home/greatbme/data_plus/Stereo Dataset',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--use-dilation',type = bool,default=True,
                    help = 'whether use dilation conv')
parser.add_argument('--use-ssp',type = bool,default=True,
                    help = 'whether enable ssp')
args = parser.parse_args()
args.cuda = (not args.no_cuda) and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_transform_list = [transforms.RandomCrop(480, 608), transforms.RandomColor(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ]
train_transform = transforms.Compose(train_transform_list)

training_transforms = albu.Compose([
        # Color augmentation
        albu.OneOf([
            albu.Compose([
                albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                # albu.RandomGamma(gamma_limit=(80, 120), p=0.01),
                albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0, p=0.5)]),
            albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5)
        ]),
        # Image quality augmentation
        albu.OneOf([
            albu.Blur(p=0.3),
            albu.MedianBlur(p=0.3, blur_limit=3),
            albu.MotionBlur(p=0.3),
        ]),
        # Noise augmentation
        albu.OneOf([
            albu.GaussNoise(var_limit=(10, 30), p=0.5),
            albu.IAAAdditiveGaussianNoise(loc=0, scale=(0.005 * 255, 0.02 * 255), p=0.5)
        ]), ], p=0.6)

if args.use_hourglass == True:
    model = stackhourglass.PSMNet(int(args.maxdisp),args.use_dilation,args.use_ssp)
else:
    model = basic.PSMNet(int(args.maxdisp),args.use_dilation,args.use_ssp)

if args.cuda:
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.cuda()

if args.loadmodel is not None:
    if Path(args.loadmodel).exists():
        print("Loading {:s} ......".format(args.loadmodel))
        state_dict = torch.load(args.loadmodel)
        step = state_dict['step']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['state_dict'])
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained model detected")
        exit()
else:
    epoch = 0
    step = 0

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
currentDT = datetime.datetime.now()
log_root = Path(args.savemodel)/"depth_estimation_training_run_{}_{}_{}_valid_{}".format(
    currentDT.month,
    currentDT.day,
    currentDT.hour, 2)
if not log_root.exists():
    log_root.mkdir()
writer = SummaryWriter(logdir=str(log_root))
print("Tensorboard visualization at {}".format(str(log_root)))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)


def train(sample):
    model.train()
    left = sample['left']  # [B, 3, H, W]
    right = sample['right']
    disp = sample['disp']  # [B, H, W]
    mask = sample['mask']
    if args.cuda:
        left, right, disp, mask = left.cuda(), right.cuda(), disp.cuda(), mask.cuda()
    mask = mask > 0.6
    b, c, h, w = disp.shape
    count = b
    for i in range(b):
        if disp[i][mask[i]].numel() < disp[i].numel() * 0.1:
            mask[i] = mask[i] < 0
            count -= 1
    if count < 1:
        return -1, -1, -1
    optimizer.zero_grad()
    if args.use_hourglass == True:
        output1, output2, output3 = model(left, right)
        b, h, w = output1.shape
        bg, c, hg, wg = disp.shape
        output1 = output1.unsqueeze(1)
        output2 = output2.unsqueeze(1)
        output3 = output3.unsqueeze(1)
        ds = hg / h
        output1 = F.upsample(output1, [hg, wg], mode='bilinear')
        output2 = F.upsample(output2, [hg, wg], mode='bilinear')
        output3 = F.upsample(output3, [hg, wg], mode='bilinear')

        output1 = torch.mul(output1, ds)  # 上采样后要乘以采样率
        output2 = torch.mul(output2, ds)
        output3 = torch.mul(output3, ds)

        # output1 = torch.squeeze(output1, 1)
        # output2 = torch.squeeze(output2, 1)
        # output3 = torch.squeeze(output3, 1)  # 输出是batch*height*width

        # depth1 = submodule.reprojection()(output1, reproj_left)
        # depth2 = submodule.reprojection()(output2, reproj_left)
        # depth3 = submodule.reprojection()(output3, reproj_left)

        # gt = rec_left_gt[0].cpu().numpy()
        # np.save('gt.npy', gt)
        # bb = bb[0].cpu().numpy()
        # np.save('b.npy', bb)
        # ff = ff[0].cpu().numpy()
        # np.save('f.npy', ff)
        # disp = output3[0].cpu().detach().numpy()
        # np.save('disparity.npy', disp)
        # z = depth3[0].cpu().detach().numpy()
        # np.save('depth.npy', z)

        # fn = torch.nn.MSELoss()
        # loss = 0.5 * F.smooth_l1_loss(depth1[mask_left], rec_left_gt[mask_left]) + 0.7 * F.smooth_l1_loss(depth2[mask_left], rec_left_gt[mask_left]) + F.smooth_l1_loss(depth3[mask_left], rec_left_gt[mask_left])

        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp[mask]) + 0.7 * F.smooth_l1_loss(output2[mask], disp[mask]) + F.smooth_l1_loss(output3[mask], disp[mask])

    elif args.model == 'basic':
        output3 = model(ds_left, ds_right)
        output3 = output3.unsqueeze(1)
        b, d, h, w = output3.shape
        bg, c, hg, wg = rec_left_gt.shape
        ds = hg / h
        output3 = F.upsample(output3, [hg, wg], mode='bilinear')
        output3 = torch.mul(output3, ds)
        output3 = torch.squeeze(output3, 1)
        depth3 = submodule.reprojection()(output3, reproj_left)
        loss = F.l1_loss(rec_left_gt[mask_left], depth3[mask_left], size_average=True)

    loss.backward()
    optimizer.step()
    # display.display_color_disparity_depth(step, writer, ds_left, output3.unsqueeze(1), depth3, is_return_img=False)
    return loss.item(), count, output3


def Test(sample):
    model.training = False
    model.eval()
    left = sample['left']  # [B, 3, H, W]
    right = sample['right']
    disp = sample['disp']  # [B, H, W]
    mask = sample['mask']

    if args.cuda:
        left, right, disp, mask = left.cuda(), right.cuda(), disp.cuda(), mask.cuda()
    mask = mask > 0.6
    b, c, h, w = disp.shape
    count = b
    for i in range(b):
        if disp[i][mask[i]].numel() < disp[i].numel() * 0.1:
            mask[i] = mask[i] < 0
            count -= 1
    if count < 1:
        return -1, -1, -1, -1, -1, -1, -1

    with torch.no_grad():
        output3 = model(left, right)
        b, h, w = output3.shape
        bg, c, hg, wg = disp.shape
        ds = hg / h
        output3 = output3.unsqueeze(1)
        output3 = F.upsample(output3, [hg, wg], mode='bilinear')
        output3 = torch.mul(output3, ds)
        # output3 = torch.squeeze(output3, 1)
        # depth3 = submodule.reprojection()(output3, reproj_left)

    output3 = output3.data.cpu()
    disp = disp.data.cpu()
    mask = mask.data.cpu()

    if len(disp[mask]) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output3[mask] - disp[mask]))  # end-point-error
        # z_compute = depth3[:, 2, :, :]
        # z_compute_bf = torch.zeros_like(z_compute)
        # z_gt = rec_left_gt[:, 2, :, :]
        # z_gt_bf = torch.zeros_like(z_gt)
        # z_mask = mask_left[:, 2, :, :]
        bad1_mask = (torch.abs(output3 - disp) > 1) & (mask)
        bad3_mask = (torch.abs(output3 - disp) > 3) & (mask)
        bad2_mask = (torch.abs(output3 - disp) > 2) & (mask)
        bad5_mask = (torch.abs(output3 - disp) > 5) & (mask)
        bad1_perc = float(torch.sum(bad1_mask)) / float(torch.sum(mask))
        bad2_perc = float(torch.sum(bad2_mask)) / float(torch.sum(mask))
        bad3_perc = float(torch.sum(bad3_mask)) / float(torch.sum(mask))
        bad5_perc = float(torch.sum(bad5_mask)) / float(torch.sum(mask))
        # rel_err = torch.mean(torch.abs(z_gt[z_mask] - z_compute[z_mask]) / z_gt[z_mask])
        # b, c, h, w = depth3.shape
        # for i in range(b):
        #     z_compute_bf[i, :, :] = z_compute[i, :, :] * bb[i] * ff[i]
        #     z_gt_bf[i, :, :] = z_gt[i, :, :] * bb[i] * ff[i]
        #
        # loss_d = torch.mean(torch.div(torch.abs(z_gt_bf[z_mask] - z_compute_bf[z_mask]), torch.mul(z_gt[z_mask], z_gt[z_mask])+0.01))
        # #print(loss)
    return loss.item(), bad1_perc, bad2_perc, bad3_perc, bad5_perc, count, output3


def adjust_learning_rate(optimizer, epoch):
    if epoch < 3:
        lr = 0.001
    else:
        lr = 0.0005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    batch_size = 8
    display_interval = 50
    validation_interval = 1

    Traindata = MyD.MyDataset(args.datapath, transform=train_transform, downsampling=2)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=Traindata, batch_size=batch_size, shuffle=True,  drop_last=False)
    # Testdata = MyD.MyDataset(args.datapath, downsampling=2, phase='test')
    # TestImgLoader = torch.utils.data.DataLoader(dataset=Testdata, batch_size=batch_size, shuffle=False, drop_last=False)
    Validationdata = MyD.MyDataset(args.datapath, downsampling=2, phase='validation')
    ValidationImgLoader = torch.utils.data.DataLoader(dataset=Validationdata, batch_size=batch_size, shuffle=False, drop_last=False)
    start_full_time = time.time()

    for epoch in range(epoch + 1, args.epochs + 1):
        print('This is %d-th epoch' % (epoch))
        mean_loss = 0
        all_count = 0
        train_step = 0
        adjust_learning_rate(optimizer, epoch)
        tq = tqdm.tqdm(total=len(TrainImgLoader) * batch_size, dynamic_ncols=True, ncols=80)

        ## training ##
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            # print('rec_right_gt.shape: ', rec_right_gt.shape)
            # print('reproj_left.shape: ', reproj_left.shape)
            # print('bb: ', type(bb), bb)
            loss, count, disparity = train(sample)
            if loss < 0:
                continue
            else:
                mean_loss = (mean_loss * all_count + loss * count)/(all_count + count)
                all_count += count
            #  print('Iter %d training loss = %.3f , time = %.2f s' % (batch_idx, loss, time.time() - start_time))
            step += 1
            train_step += 1

            tq.update(batch_size)
            tq.set_postfix(loss='avg: {:.5f}  cur: {:.5f}'.format(mean_loss, loss))
            writer.add_scalar('Training/loss',  mean_loss, step)
            if train_step % display_interval == 0:
                display.display_color_disparity(epoch, train_step, writer, sample['left'], disparity, sample['disp'], sample['mask'], is_return_img=False)
            if train_step > 1800:
                break
        tq.close()

        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'step': step,
            'state_dict': model.state_dict(),
            'train_loss': mean_loss,
        }, savefilename)
        writer.export_scalars_to_json(str(log_root / ('all_scalars_' + str(epoch) + '.json')))

        if epoch % validation_interval != 0:
           continue

        valid_mean_loss = 0.0
        all_count = 0
        val_step = 0
        mean_bad1_perc = 0
        mean_bad2_perc =0
        mean_bad3_perc=0
        mean_bad5_perc=0
        tq = tqdm.tqdm(total=len(ValidationImgLoader) * batch_size, dynamic_ncols=True, ncols=80)
        tq.set_description('Validation Epoch {}'.format(epoch))
        with torch.no_grad():
            for batch_idx, sample in enumerate(ValidationImgLoader):
                loss, bad1_perc, bad2_perc, bad3_perc, bad5_perc, count, disparity = Test(sample)
                if loss < 0:
                    continue
                else:
                    valid_mean_loss = (valid_mean_loss * all_count + loss * count) / (all_count + count)
                    mean_bad1_perc = (mean_bad1_perc * all_count + bad1_perc * count) / (all_count + count)
                    mean_bad2_perc = (mean_bad2_perc * all_count + bad2_perc * count) / (all_count + count)
                    mean_bad3_perc = (mean_bad3_perc * all_count + bad3_perc * count) / (all_count + count)
                    mean_bad5_perc = (mean_bad5_perc * all_count + bad5_perc * count) / (all_count + count)
                    all_count += count
                #  print('Iter %d training loss = %.3f , time = %.2f s' % (batch_idx, loss, time.time() - start_time))
                tq.update(batch_size)
                tq.set_postfix(loss='Valid_avg: {:.5f}  Valid_cur: {:.5f}'.format(valid_mean_loss, loss))
                # tq.set_postfix(disparity_loss='Valid_avg: {:.5f}   Valid_cur: {:.5f}'.format(valid_mean_disparity_loss, loss_d))
                val_step += 1
                writer.add_scalar('Validation/epoch{}disparity_EPE'.format(epoch), loss, val_step)
                writer.add_scalar('Validation/epoch{}_bad2_perc'.format(epoch), bad1_perc, val_step)
                if val_step % display_interval == 0:
                    display.display_color_disparity(epoch, val_step, writer, sample['left'], disparity, sample['disp'], sample['mask'], phase='validation', is_return_img=False)

            writer.add_scalar('Validation/disparity_EPE', valid_mean_loss, epoch)
            writer.add_scalar('Validation/bad1_perc', mean_bad1_perc, epoch)
            writer.add_scalar('Validation/bad2_perc', mean_bad2_perc, epoch)
            writer.add_scalar('Validation/bad3_perc', mean_bad3_perc, epoch)
            writer.add_scalar('Validation/bad5_perc', mean_bad5_perc, epoch)
        tq.close()

    #  print('epoch %d mean training loss = %.3f' % (epoch, mean_loss))

        # SAVE


    print('full training time = %.2f HR' % ((time.time() - start_full_time)/3600))