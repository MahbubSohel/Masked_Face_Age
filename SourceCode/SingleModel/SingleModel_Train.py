import os
import shutil
import torch.backends.cudnn as cudnn
import numpy as np
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim
import torchvision.models as models
from skimage import io, transform
from scipy import ndimage as sio
from datetime import datetime
from torch.optim import optimizer
from torch.autograd import variable
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

warnings.filterwarnings("ignore")
plt.ion()

L_Rate = 0.001
num_epochs = 12
best_acc1 = 0
frequency = 1000


# Image Processing ( Transformation)


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, age, group = sample['images'], sample['ages'], sample['group']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'images': img, 'ages': age, 'group': group}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
       
        image, age, group = sample['images'], sample['ages'], sample['group']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        return {'images': image, 'ages': age, 'group': group}


class ToTensor(object):

    def __call__(self, sample):
        image, age, group = sample['images'], sample['ages'], sample['group']
        image = image.transpose((2, 0, 1))
        images = torch.from_numpy(image)
        return images, age, group


# Data Preparation


class AgePredictionData(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.face_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.face_frame.ix[idx, 0])
        image = sio.imread(img_name, mode='RGB')
        age = self.face_frame.ix[idx, 1].astype('float')
        groupAge = self.face_frame.ix[idx, 2].astype('float')
        sample = {'images': image, 'ages': age, 'group': groupAge}
        if self.transform:
            sample = self.transform(sample)
        return sample


class FocalLoss(nn.Module):
 
    def __init__(self, gamma=0, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
 
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        loss = self.alpha * loss
        return loss.mean()


use_gpu = torch.cuda.is_available()


# Main Function


def main():
    global frequency, best_acc1

    df=pd.read_csv('/home/cviplab/MaskAge/SourceCode/FaceMask_OneModel/3G_v2/3Gv2_Age_Dist.csv')
    wghts= df['Weights']
    nwghts= np.array(wghts)
    weights=torch.from_numpy(nwghts)
    weights_var = weights.float().cuda()

    train_data = AgePredictionData(csv_file='/home/cviplab/MaskAge/file_paths/3G_v2/group_all_train.csv',
                                   root_dir='/home/cviplab/MaskAge/DataSet/CropMask/',
                                   transform=transforms.Compose([
                                       Rescale(256),
                                       RandomCrop(224),
                                       ToTensor()
                                   ]))

    valid_data = AgePredictionData(csv_file='/home/cviplab/MaskAge/file_paths/3G_v2/group_all_val.csv',
                                   root_dir='/home/cviplab/MaskAge/DataSet/CropMask/',
                                   transform=transforms.Compose([
                                       Rescale(256),
                                       RandomCrop(224),
                                       ToTensor()
                                   ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                               shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(valid_data, batch_size=32,
                                             shuffle=True, num_workers=4)

   
    #model = models.squeezenet1_1(pretrained=True)
    #model.classifier[1] = nn.Conv2d(512, 101, kernel_size=(1,1), stride=(1,1))
    #model.cuda()

    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 101)
    model.cuda()

    #model = models.shufflenet_v2_x1_0(pretrained=True)
    #model.fc = nn.Linear(1024, 101)
    #model.cuda()

    #model = models.resnet50(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs,101)
    #model.cuda()

    #model = models.mobilenet_v2(pretrained=True)
    #model.classifier[1] = nn.Linear(1280, 101)
    #model.cuda()

    #model = models.wide_resnet50_2(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs,101)
    #model.cuda()

    
    #criterion = FocalLoss(gamma=2,alpha=0.25)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = nn.CrossEntropyLoss(weights_var).cuda()

    #optimizer = torch.optim.Adam(model.parameters(), L_Rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), L_Rate, momentum=0.9, weight_decay=1e-4)
    cudnn.benchmark = True
  
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    start_time = time.time()

    for epoch in range(num_epochs):
        #adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion1, optimizer, epoch)
        #acc1 = validate(val_loader, model, criterion, epoch)
        acc1,val_loss = validate(val_loader, model, criterion, epoch)
        scheduler.step(val_loss)

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint(model.state_dict(), is_best, epoch)

    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print("Duration(hrs):")
    print(duration)


def train(train_loader, model, criterion1, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    for i, (inp, target, groupTarget) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input1 = torch.FloatTensor()
        input1 = inp
        if use_gpu:
            input_var = torch.autograd.Variable(input1.float().cuda())
            target_var = torch.autograd.Variable(target.long().cuda())

        else:
            input_var = torch.autograd.Variable(input1.float())
            target_var = torch.autograd.Variable(target.long())
        output_var = model(input_var)
        loss = criterion1(output_var, target_var)

        acc1, acc5 = accuracy(output_var.data, target_var.data, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1.item(), inp.size(0))
        top5.update(acc5.item(), inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % frequency == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'accuracy@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                      loss=losses, top1=top1, top5=top5))

    f = open('Train_OM_DN_3Gv2_LW_e12.txt', 'a')
    f.write('Epoch: {0}\t''Loss: {loss.avg:.4f}\t''acc@1 {top1.avg:.3f}\t''acc@5 {top5.avg:.3f}\n'.format(epoch,
                                                                                                          loss=losses,
                                                                                                          top1=top1,
                                                                                                          top5=top5))
    f.close()


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (inp, target, groupTarget) in enumerate(val_loader):
        inp = inp.cuda()
        input1 = torch.FloatTensor()
        input1 = inp
        if use_gpu:
            input_var = torch.autograd.Variable(input1.float().cuda())
            target_var = torch.autograd.Variable(target.long().cuda())

        else:
            input_var = torch.autograd.Variable(input1.float())
            target_var = torch.autograd.Variable(target.long())

        output_var = model(input_var)
        loss = criterion(output_var, target_var)

        acc1, acc5 = accuracy(output_var.data, target_var.data, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1.item(), inp.size(0))
        top5.update(acc5.item(), inp.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Accuracy@1 {top1.avg:.3f} Accuracy@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    f = open('Val_OM_DN_3Gv2_LW_e12.txt', 'a')
    f.write('Epoch: {0}\t''Loss: {loss.avg:.4f}\t''acc@1 {top1.avg:.3f}\t''acc@5 {top5.avg:.3f}\n'.format(epoch,
                                                                                                          loss=losses,
                                                                                                          top1=top1,
                                                                                                          top5=top5))
    f.close()
    return top1.avg,losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, epoch, filename='ckp_OM_DN_3Gv2_LW_e12.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'OM_DN_3Gv2_LW_e12.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = L_Rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
