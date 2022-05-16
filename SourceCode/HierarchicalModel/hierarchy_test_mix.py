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

# ********************************************************************************************************************
# Main function
# ********************************************************************************************************************
def main():
    
    # Model for group Age
    
    #grpModel = models.squeezenet1_1(pretrained=False)
    #grpModel.classifier[1] = nn.Conv2d(512, 3, kernel_size=(1,1), stride=(1,1)) 
    #grpModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/3G_v2/3Gv2_sqNet_LW_grAge.pth.tar'))
    #grpModel.cuda()

    grpModel = models.shufflenet_v2_x1_0(pretrained=False)
    grpModel.fc = nn.Linear(1024,3)
    grpModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/3G_v2/3Gv2_shflN_LW_grAge.pth.tar'))
    grpModel.cuda()
    
    #grpModel = models.resnet50(pretrained=False)
    #num_ftrs = grpModel.fc.in_features
    #grpModel.fc = nn.Linear(num_ftrs,3)
    #grpModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/3G_v2/3Gv2_RN_LW_grAge.pth.tar'))
    #grpModel.cuda()

    # Model for group 0
     
    #zeroModel = models.squeezenet1_1(pretrained=False)
    #zeroModel.classifier[1] = nn.Conv2d(512, 16, kernel_size=(1,1), stride=(1,1)) 
    #zeroModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_0_sqNet_LW.pth.tar'))
    #zeroModel.cuda() 

    zeroModel = models.shufflenet_v2_x1_0(pretrained=False)
    zeroModel.fc = nn.Linear(1024,16)
    zeroModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_0_shflNet_LW.pth.tar'))
    zeroModel.cuda()

    #zeroModel = models.resnet50(pretrained=False)
    #num_ftrs = zeroModel.fc.in_features
    #zeroModel.fc = nn.Linear(num_ftrs,16)
    #zeroModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_0_RN_LW.pth.tar'))
    #zeroModel.cuda() 
 

    # Model for group 1 
    '''     
    oneModel = models.squeezenet1_1(pretrained=False)
    oneModel.classifier[1] = nn.Conv2d(512, 49, kernel_size=(1,1), stride=(1,1)) 
    oneModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_sqNet_lrs.pth.tar'))
    oneModel.cuda() 
    '''
    oneModel = models.shufflenet_v2_x1_0(pretrained=False)
    oneModel.fc = nn.Linear(1024,49)
    oneModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_shflNet_lrs.pth.tar'))
    oneModel.cuda() 

    #oneModel = models.resnet50(pretrained=False)
    #num_ftrs = oneModel.fc.in_features
    #oneModel.fc = nn.Linear(num_ftrs,49)
    #oneModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_1_RN_lrs.pth.tar'))
    #oneModel.cuda() 
    
    # Model for group 2 
    '''    
    twoModel = models.squeezenet1_1(pretrained=False)
    twoModel.classifier[1] = nn.Conv2d(512, 36, kernel_size=(1,1), stride=(1,1)) 
    twoModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_2_sqNet_LW.pth.tar'))
    twoModel.cuda() 
    '''
    twoModel = models.shufflenet_v2_x1_0(pretrained=False)
    twoModel.fc = nn.Linear(1024,36)
    twoModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_2_shflNet_LW.pth.tar'))
    twoModel.cuda() 

    #twoModel = models.resnet50(pretrained=False)
    #num_ftrs = twoModel.fc.in_features
    #twoModel.fc = nn.Linear(num_ftrs,36)
    #twoModel.load_state_dict(torch.load('/home/cviplab/MaskAge/SourceCode/groupage/Hierarchy/3G_v2/3Gv2_2_RN_LW.pth.tar'))
    #twoModel.cuda() 


    # define loss function (criterion) and optimizer
    #criterion = FocalLoss(gamma=2,alpha=0.25)

    criterion = nn.CrossEntropyLoss().cuda()                                                        
    cudnn.benchmark = False

    #Loading the data through csv and preparing dataset   
    transformed_test_dataset = AgeEstimationDataset(csv_file='/home/cviplab/MaskAge/file_paths/3G_v2/group_all_test.csv',
                                           root_dir='/home/cviplab/MaskAge/DataSet/CropMask/',
                                           transform=transforms.Compose([
                                           Rescale(256),
                                           RandomCrop(224),                                           
                                           ToTensor()                                                                                                             
                                           ]))
        
    # Loading dataset into dataloader
    test_loader =  torch.utils.data.DataLoader(transformed_test_dataset, batch_size=1,
                                               shuffle=False, num_workers=8)

    
    
    grpModel.eval()
    zeroModel.eval()
    oneModel.eval()
    twoModel.eval()

    for i, (input, target, groupTarget) in enumerate(test_loader):
        input= input.cuda()  
        input1= torch.FloatTensor()
        input1=input
        if use_gpu:         
           input_var = torch.autograd.Variable(input1.float().cuda())
           gtAge = torch.autograd.Variable(target.long().cuda())
           grptarget_var = torch.autograd.Variable(groupTarget.long().cuda())
        else:
           input_var = torch.autograd.Variable(input1.float())
           gtAge = torch.autograd.Variable(target.long())
           grptarget_var = torch.autograd.Variable(groupTarget.long())

        # compute output
        output = grpModel(input_var)
        softprob = torch.nn.functional.softmax(output.data, dim=1)
        _, pred = softprob.topk(1, 1, True, True)
        predGrpAge = pred.item()
        target = gtAge.data
        TA = gtAge.data.item()
        #print('Target Age: {0}\n'.format(target.item()))
        #print('predicted group: {0}\n'.format(pred.item()))
        

        if predGrpAge == 0:
           zeroOutput = zeroModel(input_var)
           zeroProb = torch.nn.functional.softmax(zeroOutput, dim=1)
           prob,label= torch.topk(zeroProb, 16)
           total= prob.mul(label.float())
           age= total.sum()
           diff = torch.abs(int(age) - target)
           f = open('Err_hrchy_RNshflNet_LW.txt', 'a')
           f.write('{0}\n'.format(diff.item()))
           f.close()
           
        if predGrpAge == 1:
           oneOutput = oneModel(input_var)
           oneProb = torch.nn.functional.softmax(oneOutput, dim=1)
           prob,label= torch.topk(oneProb, 49)
           total= prob.mul(label.float()+16)
           age= total.sum()
           diff = torch.abs(int(age) - target)
           f = open('Err_hrchy_RNshflNet_LW.txt', 'a')
           f.write('{0}\n'.format(diff.item()))
           f.close()
           
        if predGrpAge == 2:
           twoOutput = twoModel(input_var)
           twoProb = torch.nn.functional.softmax(twoOutput, dim=1)
           prob,label= torch.topk(twoProb, 36)
           total= prob.mul(label.float()+65)
           age= total.sum()
           diff = torch.abs(int(age) - target)
           f = open('Err_hrchy_RNshflNet_LW.txt', 'a')
           f.write('{0}\n'.format(diff.item()))
           f.close()
           

if __name__ == '__main__':
   main()

