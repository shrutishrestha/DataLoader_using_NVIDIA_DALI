import os
import csv
import cv2
import torch
from datetime import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cProfile, pstats
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

#for neural network
from torchvision import datasets, transforms
    
    
def make(folderpathlist):
    for folderpath in folderpathlist:
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
            
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512,512)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(saturation=1.2, brightness=(0, 1), contrast=(0, 1)),
            transforms.GaussianBlur(kernel_size=(5,5)),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0].strip())
        
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

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
        device = ("cuda" if torch.cuda.is_available() else "cpu" )    
        if args["sbatch"]:
            make([args["snapshot_directory"],args["snapshot_model_dir"], args["result_directory"], args["result_model_dir"]])
            

        # for dataset and dataloader
        traindataset = CustomImageDataset(transform = True, annotations_file = args["train_file"], img_dir = args["train_dir"])     
        valdataset = CustomImageDataset(annotations_file = args["val_file"], img_dir = args["val_dir"])     
        model = ResNet34().to(device)
        
        model = torch.nn.DataParallel(model)

        criterion = torch.nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

        trainDataLoader = DataLoader(traindataset, batch_size=args["train_batch_size"], shuffle=False, num_workers=args["num_threads"], pin_memory=True)
        valDataLoader = DataLoader(valdataset, batch_size=args["val_batch_size"], shuffle=False, num_workers=args["num_threads"], pin_memory=True)
        
        
        best_ckpt = {"epoch":-1, "current_val_metric":0, "model":model.state_dict()}

        epoch = args["start_iters"]

        epoch_list = []
        datetime_list = []
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        lenTrainDataLoader = len(trainDataLoader)
        lenValDataLoader = len(valDataLoader)
        
        for epoch in range(args["max_epochs"]):
            print("epoch",epoch, datetime.now())
            loss_sum = 0
            acc_sum = 0

            for i, batch in enumerate(trainDataLoader):

                image, label = batch
                
                image = image.to(device)
                label = label.to(device)
                label = label.to(torch.float)

                optimizer.zero_grad()
                logits = model(image).flatten()

                pred_probab = nn.Sigmoid()(logits)
                y_pred = (pred_probab>0.5)
                
                #forward pass
                loss = criterion(pred_probab, label)
                #backward and optimize
                loss.backward()
                optimizer.step()

                acc = accuracy_score(label.cpu(), y_pred.cpu())
                loss_sum += loss.item()
                acc_sum += acc.item()

            train_loss = round(loss_sum/lenTrainDataLoader,6)
            train_acc = round(acc_sum/lenTrainDataLoader,6)       


            model.eval()
            loss_sum = 0
            acc_sum = 0
            with torch.no_grad():
                for i, batch in enumerate(valDataLoader):
                    image, label = batch
                    image = image.to(device)
                    label = label.to(device)
                    label = label.to(torch.float)

                    logits = model(image).flatten()
                    pred_probab = nn.Sigmoid()(logits)
                    y_pred = (pred_probab>0.5)

                    #forward pass
                    loss = criterion(pred_probab, label)

                    #backward and optimize
                    acc = accuracy_score(label.cpu(), y_pred.cpu())
                    loss_sum += loss.item()
                    acc_sum += acc.item()


            val_loss = round(loss_sum/lenValDataLoader,6)
            val_acc = round(acc_sum/lenValDataLoader,6)  
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
        print("all done",datetime.now())

        if args["sbatch"]:

            header = ["epoch", "epoch start datetime", "train_loss", "train_acc", "val_loss", "val_acc"]
            rows = zip(epoch_list, datetime_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list)

            with open(os.path.join(args["result_model_dir"], "results.csv"), 'w') as csvfile: 
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(header) 
                for row in rows:
                    csvwriter.writerow(row) 

            torch.save(current_ckpt, args["current_checkpoint_fpath"])
            torch.save(best_ckpt, args["best_checkpoint_fpath"])

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
args = {}
fake = False
cprof = True
args["sbatch"] = False
args["max_epochs"] = 4

args["num_threads"] = 1

args["seed"] = 42
args["train_batch_size"] = 8
args["val_batch_size"] = 8
args["learning_rate"] = 0.01
args["start_iters"] = 0
args["set_affinity"] = True

if fake:
    args["train_file"]= "catdogs/fake/bc/train_all.csv"
    args["val_file"]= "catdogs/fake/bc/val_all.csv"
    args["train_dir"] = "catdogs/fake/bc/train_all"
    args["val_dir"] = "catdogs/fake/bc/val_all"
else:
    args["train_file"]= "catdogs/real/bc/train_all.csv"
    args["val_file"]= "catdogs/real/bc/val_all.csv"
    args["train_dir"] = "catdogs/real/bc/train_all"
    args["val_dir"] = "catdogs/real/bc/val_all"

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
