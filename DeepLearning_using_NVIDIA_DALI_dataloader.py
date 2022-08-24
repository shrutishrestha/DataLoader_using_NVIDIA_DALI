import csv
import cv2
import glob
import math
import nvtx
import torch
import shutil
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import cProfile, pstats
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

#for neural network
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



import torch
import torch.nn as nn
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

def make(folderpathlist):
    for folderpath in folderpathlist:
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
            
@pipeline_def
def custom_pipeline(files, image_dir, train):
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    images = fn.resize(images, resize_x=512, resize_y=512)
    if train:
        mt = fn.transforms.rotation(angle = fn.random.uniform(range=(-45, 45)))
        images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
        images = fn.brightness_contrast(images, brightness=fn.random.uniform(range=(0,1)), contrast= fn.random.uniform(range=(0,1)))
        images = fn.saturation(images, saturation = fn.random.uniform(range=(0,1)))
        images = fn.gaussian_blur(images, window_size=[5,5])
    images = fn.transpose(images, perm=(2,0,1))
    images = fn.normalize(images)
    return images, labels.gpu()

class ExternalInputGpuIterator(DALIGenericIterator):
    def __init__(self, pipelines, batch_size, files, last_batch_policy,  last_batch_padded, auto_reset=True):
        super().__init__(pipelines=pipelines, last_batch_policy=last_batch_policy,  last_batch_padded = last_batch_padded, auto_reset=auto_reset, output_map=['images', 'labels'])
        self.files = files
        self.batch_size = batch_size
        self.data_set_len = len(self.files)
        self.n = self.data_set_len
        

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __len__(self):
        return self.data_set_len

    def __next__(self):

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        else:
            out = super().__next__()
            images = out[0]['images']
            labels = out[0]['labels'] 

            q = (self.n - self.i) // self.batch_size
            mod = (self.n - self.i) % self.batch_size
            if q>0:
                self.i = self.i + self.batch_size
            else: 
                self.i = self.i + mod

            return (images, labels)
   
    next = __next__
    
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding=1, bias= False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()

        # enters from block 2
        if stride != 1 or in_planes != self.expansion * planes: #stride != 1, not first layer //// in_planes - 64, planes x self.expansion = 64 x 1 //64,64
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size = 1,stride = stride, bias= False), nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):

       # first layer
        h_x = self.conv1(x)
        h_x = self.bn1(h_x)
        h_x = self.relu(h_x)

       # second layer
        h_x = self.conv2(h_x)
        h_x = self.bn2(h_x)

       # adding identity / shortcut

        h_x += self.shortcut(x)
        y = self.relu(h_x)

        return y
    
class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(Resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer1 = self.make_layer(block, 64, num_blocks[0], first_stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], first_stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], first_stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], first_stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def make_layer(self, block, planes, num_blocks, first_stride):
        other_strides = [1]
        strides = [first_stride] + other_strides * (num_blocks - 1)       # first stride is for size reduction, other strides are 1 incase of 18 and 34

        layers = [] 
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion       # for resnet 18 and 34, inchannel == outchannel, so expansion = 1
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        
        output = output.reshape(output.shape[0], -1)
        
        output = self.linear(output)
        
        return output
    
def ResNet34():
    return Resnet(BasicBlock, [3, 4, 6, 3], num_classes=1)
    
    
def main(args):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # if args["sbatch"]:
        #     make([args["snapshot_directory"],args["snapshot_model_dir"], args["result_directory"], args["result_model_dir"]]) 

        model = ResNet34().to(device)
        
        criterion = torch.nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)


        train_file_list = os.listdir(os.path.join(args["train_dir"],"1"))+ os.listdir(os.path.join(args["train_dir"],"0"))
        
        val_file_list = os.listdir(os.path.join(args["val_dir"],"1"))+ os.listdir(os.path.join(args["val_dir"],"0"))
        
        pipe_train_gpu = custom_pipeline(batch_size=args["train_batch_size"], device_id=args["device_id"],
                                         num_threads=args["num_threads"], files= args["train_file"], 
                                         set_affinity=args["set_affinity"], image_dir = args["train_dir"], train =True)

        dali_train_iter = ExternalInputGpuIterator(pipe_train_gpu, batch_size=args["train_batch_size"], 
                                                   last_batch_policy=LastBatchPolicy.PARTIAL,  last_batch_padded = False, files=train_file_list)
        
        pipe_val_gpu = custom_pipeline(batch_size=args["val_batch_size"], device_id=args["device_id"],
                                         num_threads=args["num_threads"], files= args["val_file"], 
                                         set_affinity=args["set_affinity"], image_dir = args["val_dir"], train=False)

        dali_val_iter = ExternalInputGpuIterator(pipe_val_gpu, batch_size=args["val_batch_size"], 
                                                   last_batch_policy=LastBatchPolicy.PARTIAL,  last_batch_padded = False, files=val_file_list)
        
        best_ckpt = {"epoch":-1, "current_val_metric":0, "model":model.state_dict()}

        epoch = args["start_iters"]

        epoch_list = []
        datetime_list = []
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        
        
        for epoch in range(args["max_epochs"]):
            print("epoch", epoch, datetime.now())
            loss_sum = 0
            acc_sum = 0


            for i, (images, labels) in enumerate(dali_train_iter):
                images = images.to(torch.float32)
                labels = labels.to(torch.float32)

                optimizer.zero_grad()
                logits = model(images)

                pred_probab = nn.Sigmoid()(logits)
                y_pred = (pred_probab>0.5)
                
                #forward pass
                loss = criterion(pred_probab, labels)
                #backward and optimize
                loss.backward()
                optimizer.step()

                acc = accuracy_score(labels.cpu(),y_pred.cpu())
                loss_sum += loss.item()
                acc_sum += acc.item()
            
            train_loss = round(loss_sum/len(dali_train_iter),6)
            train_acc = round(acc_sum/len(dali_train_iter),6)       


            model.eval()
            loss_sum = 0
            acc_sum = 0
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(dali_val_iter):              
                    images = images.to(torch.float32)
                    labels = labels.to(torch.float32)
                    logits = model(images)
                    pred_probab = nn.Sigmoid()(logits)
                    y_pred = (pred_probab>0.5)

                    #forward pass
                    loss = criterion(pred_probab, labels)

                    #backward and optimize
                    acc = accuracy_score(labels.cpu(),y_pred.cpu())
                    loss_sum += loss.item()
                    acc_sum += acc.item()


            val_loss = round(loss_sum/len(dali_val_iter),6)
            val_acc = round(acc_sum/len(dali_val_iter),6)  
            
            dali_train_iter.reset()
            dali_val_iter.reset()
            print(train_loss, val_loss, train_acc, val_acc)

            current_ckpt = {"epoch":epoch, "current_val_metric":val_acc, "model":model.state_dict()}

            if best_ckpt["current_val_metric"] < val_acc:
                best_ckpt = {"epoch":epoch, "current_val_metric":val_acc, "model":model.state_dict()}

            epoch_list.append(epoch)
            datetime_list.append(datetime.now())
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            scheduler.step(val_loss)

#         if args["sbatch"]:

#             header = ["epoch", "epoch start datetime", "train_loss", "train_acc", "val_loss", "val_acc"]
#             rows = zip(epoch_list, datetime_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list)

#             with open(os.path.join(args["result_model_dir"], "results.csv"), 'w') as csvfile: 
#                 csvwriter = csv.writer(csvfile) 
#                 csvwriter.writerow(header) 
#                 for row in rows:
#                     csvwriter.writerow(row) 

#             torch.save(current_ckpt, args["current_checkpoint_fpath"])
#             torch.save(best_ckpt, args["best_checkpoint_fpath"])

args = {}
fake = False
cprof = True
args["sbatch"] = False
args["train_batch_size"]= 4
args["val_batch_size"] = 4
args["learning_rate"] = 0.01
args["device_id"] = 0
args["num_threads"] = 1
args["dataloader_output"] = 2
args["set_affinity"] = True
args["start_iters"] = 0
args["max_epochs"] = 4
args["seed"] = 42

if fake:
    args["train_file"]="catdogs/fake/dali/train_all_withoutlabels.csv"
    args["val_file"]="catdogs/fake/dali/val_all_withoutlabels.csv"
    args["train_dir"] = "catdogs/fake/dali/train"
    args["val_dir"] = "catdogs/fake/dali/val"
else:
    args["train_file"]="catdogs/real/dali/train_all_withoutlabels.csv"
    args["val_file"]="catdogs/real/dali/val_all_withoutlabels.csv"
    args["train_dir"] = "catdogs/real/dali/train"
    args["val_dir"] = "catdogs/real/dali/val"

job_id = os.getenv('SLURM_JOB_ID')
args["snapshot_directory"] = "/scratch/"+str(job_id)+"/snapshots/"
args["snapshot_model_dir"] = os.path.join(args["snapshot_directory"])
args["best_checkpoint_fpath"] = "/scratch/"+str(job_id)+"/snapshots/"+"/best_checkpoint.pth"
args["current_checkpoint_fpath"] = "/scratch/"+str(job_id)+"/snapshots/"+"/current_checkpoint.pth"
args["result_directory"] = "/scratch/"+str(job_id)+"/results/"
args["result_model_dir"] = os.path.join(args["result_directory"])

if cprof:
    profiler = cProfile.Profile()
    profiler.enable()
    main(args)
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
    stats.print_stats(60)
else:
    main(args)
