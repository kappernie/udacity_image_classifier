import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import fsupport
#programmer : Appau Ernest Kofi Mensah
#reference on code for train.py  ,predict.py  and fsupport.py are from Medium blog, stackoverflow site, pytorch blog  and google 
 
arpass = argparse.ArgumentParser(description='Train.py')


arpass.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
arpass.add_argument('--gpu', dest="gpu", action="store", default="gpu")
arpass.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
arpass.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
arpass.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
arpass.add_argument('--epochs', dest="epochs", action="store", type=int, default=12)
arpass.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
arpass.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

vpass = arpass.parse_args()

imageset_path =vpass .data_dir
path = vpass .save_dir
lr = vpass .learning_rate
structure = vpass.arch
dropout = vpass .dropout
hidden_layer1 =vpass .hidden_units
power = vpass .gpu
epochs =vpass .epochs


trainset_dataloader, testset_dataloader, validationset_dataloader = fsupport.load_data(imageset_path)


model, optimizer, criterion = fsupport.nn_setup(structure,dropout,hidden_layer1,lr,power)


fsupport.train_and_save_network(model, optimizer, criterion, epochs, 20, trainset_dataloader , power)


print(" Model is trained and saved") 
