import argparse
import os
import shutil
import time
import pandas as pd
from skimage import io, transform
from scipy import ndimage as sio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torch.autograd import Variable
from collections import OrderedDict
from torchsummary import summary

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


plt.ion()   # interactive mode



# ********************************************************************************************************************
# Image Processing
# ********************************************************************************************************************

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

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
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

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

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'images': image, 'ages': age, 'group': group}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, age, group = sample['images'], sample['ages'], sample['group']
        image = image.transpose((2, 0, 1))
        images = torch.from_numpy(image)
        return images, age, group
        
# ********************************************************************************************************************
# Loading csv file
# ********************************************************************************************************************

class AgeEstimationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        class2idx = {
             16:0,17:1,18:2,19:3,20:4,21:5,22:6,23:7,24:8,25:9,26:10,
             27:11,28:12,29:13,30:14,31:15,32:16,33:17,34:18,35:19,36:20,
             37:21,38:22,39:23,40:24,41:25,42:26,43:27,44:28,45:29,46:30,
             47:31,48:32,49:33,50:34,51:35,52:36,53:37,54:38,55:39,56:40,
             57:41,58:42,59:43,60:44,61:45,62:46,63:47,64:48}

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

# ********************************************************************************************************************
# Main function
# ********************************************************************************************************************
def main():
    global print_freq,best_prec1
    print_freq=100

    '''
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,49)
    model.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_RN_lrs_e20.pth.tar'))
    model.cuda()
    
    model = models.wide_resnet50_2(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,49)
    model.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_WRN_lrs_e20.pth.tar'))
    model.cuda()
    
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 49)
    model.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_DN_lrs_e30.pth.tar'))
    model.cuda()
    
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 49)
    model.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_MN_lrs_e20.pth.tar'))
    model.cuda()
    
    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, 49, kernel_size=(1,1), stride=(1,1)) 
    model.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_sqNet_lrs_e50.pth.tar'))
    model.cuda()
    '''
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Linear(1024, 49)
    model.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_shflNet_lrs.pth.tar'))
    model.cuda()
    
     
   
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
                                                            
    cudnn.benchmark = False

    #Loading the data through csv and preparing dataset   
    transformed_test_dataset = AgeEstimationDataset(csv_file='/home/cviplab/MaskAge/file_paths/3G_v2/group_1_test.csv',
                                           root_dir='/home/cviplab/MaskAge/DataSet/CropMask/',
                                           transform=transforms.Compose([
                                           Rescale(256),
                                           RandomCrop(224),                                           
                                           ToTensor() 
                                           ]))
        
    # Loading dataset into dataloader
    test_loader =  torch.utils.data.DataLoader(transformed_test_dataset, batch_size=1,
                                               shuffle=False, num_workers=8)


    start_time= time.time()
    
    #Test the model
    prec1 = test(test_loader, model, criterion)
    
    end_time = time.time() 
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)

use_gpu = torch.cuda.is_available()

# ********************************************************************************************************************
# Test the model
# ********************************************************************************************************************
def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

 
    model.eval()

    end = time.time()
    for i, (input, target, groupTarget) in enumerate(val_loader):
        input= input.cuda()
                
        input1= torch.FloatTensor()
        input1=input
        if use_gpu:         
           input_var = torch.autograd.Variable(input1.float().cuda())
           target_var = torch.autograd.Variable(target.long().cuda())

        else:
           input_var = torch.autograd.Variable(input1.float())
           target_var = torch.autograd.Variable(target.long())


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
      
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    
    return top1.avg



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    predicted = AverageMeter()
    
    softprob = torch.nn.functional.softmax(output, dim=1)
    target = target + 16
    # print('Target Age:{0}'.format(target.item()))

    _, pred = output.topk(maxk, 1, True, True)
    prob,label= torch.topk(softprob, 49)
    # print(label)
    # labelupdate = label.float() + 16
    # print(labelupdate)
    total= prob.mul(label.float()+16)
    age= total.sum()
    # print('Predicted Age: {0}\n'.format(age.item()))
    # print(age)
    diff = torch.abs(int(age) - target)

    # print('Age Difference: {0}\n'.format(diff.item()))
    
    
    #Writing the error value to a text file
    f = open('Err_3Gv2_1_sqNet_lrs_e50.txt', 'a')
    #f.write('Target: {0}, predicted: {1}, Difference: {2}\n'.format(target.item(), age.item(), diff.item()))
    f.write('{0}\n'.format(diff.item()))
    f.close()
    
    pred = pred.t()    
    predicted=pred
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)      
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class AverageMeter(object):
    """Computes and stores the average and current value"""
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
  

if __name__ == '__main__':
   main()
