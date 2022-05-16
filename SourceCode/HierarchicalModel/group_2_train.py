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
frequency = 500


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
        class2idx = {
             65:0,66:1,67:2,68:3,69:4,70:5,71:6,72:7,73:8,74:9,75:10,
             76:11,77:12,78:13,79:14,80:15,81:16,82:17,83:18,84:19,85:20,
             86:21,87:22,88:23,89:24,90:25,91:26,92:27,93:28,94:29,95:30,
             96:31,97:32,98:33,99:34,100:35}

        idx2class = {v: k for k, v in class2idx.items()}
        self.face_frame['Age'].replace(class2idx, inplace=True)

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



use_gpu = torch.cuda.is_available()


# Main Function


def main():
    global frequency, best_acc1

    df=pd.read_csv('/home/cviplab/MaskAge/SourceCode/group_2/3G_v2/3Gv2_2_dist.csv')
    wghts= df['Weights']
    nwghts= np.array(wghts)
    weights=torch.from_numpy(nwghts)
    weights_var = weights.float().cuda()

    train_data = AgePredictionData(csv_file='/home/cviplab/MaskAge/file_paths/3G_v2/group_2_train.csv',
                                   root_dir='/home/cviplab/MaskAge/DataSet/CropMask/',
                                   transform=transforms.Compose([
                                       Rescale(256),
                                       RandomCrop(224),
                                       ToTensor()
                                   ]))

    valid_data = AgePredictionData(csv_file='/home/cviplab/MaskAge/file_paths/3G_v2/group_2_val.csv',
                                   root_dir='/home/cviplab/MaskAge/DataSet/CropMask/',
                                   transform=transforms.Compose([
                                       Rescale(256),
                                       RandomCrop(224),
                                       ToTensor()
                                   ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16,
                                               shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(valid_data, batch_size=16,
                                             shuffle=True, num_workers=4)


    #model = models.squeezenet1_1(pretrained=True)
    #model.classifier[1] = nn.Conv2d(512, 36, kernel_size=(1,1), stride=(1,1))

    #model = models.shufflenet_v2_x1_0(pretrained=True)
    #model.fc = nn.Linear(1024, 36)

    #model = models.resnet50(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs,36)
    #model.cuda()
    
    #model = models.wide_resnet50_2(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs,36)
    #model.cuda()

    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 36)
    model.cuda()

    #model = models.mobilenet_v2(pretrained=True)
    #model.classifier[1] = nn.Linear(1280, 36)
    #model.cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = nn.CrossEntropyLoss(weights_var).cuda()

    optimizer = torch.optim.SGD(model.parameters(), L_Rate, momentum=0.9, weight_decay=1e-4)
    cudnn.benchmark = True

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)    
    start_time = time.time()

    for epoch in range(num_epochs):
        train(train_loader, model, criterion1, optimizer, epoch)
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

    f = open('Train_3Gv2_2_DN_LW.txt', 'a')
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

    f = open('Val_3Gv2_2_DN_LW.txt', 'a')
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


def save_checkpoint(state, is_best, epoch, filename='ckp_3Gv2_2_DN_LW.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '3Gv2_2_DN_LW.pth.tar')

if __name__ == '__main__':
    main()
