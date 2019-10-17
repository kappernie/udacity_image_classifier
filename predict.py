import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import fsupport
#programmer : Appau Ernest Kofi Mensah
#reference on code for train.py  ,predict.py  and fsupport.py are from Medium blog, stackoverflow site, pytorch blog  and google 


arpass= argparse.ArgumentParser( description='predict-file')


arpass.add_argument('input_img', default='/home/workspace/ImageClassifier/flowers/test/23/image_03454.jpg',nargs='*', action="store", type = str)
arpass.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
arpass.add_argument('--topk', default=5, dest="topk", action="store", type=int)
arpass.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
arpass.add_argument('--gpu', default="gpu", action="store", dest="gpu")

vpass = arpass.parse_args()

path_image = vpass.input_img
number_of_outputs =vpass.topk
power = vpass.gpu
input_img = vpass.input_img
path = vpass.checkpoint



trainset_dataloader, testset_dataloader, validationset_dataloader = fsupport.load_data()


probabilities_of_predictions, predicted_classes=fsupport.load_and_predict(path,input_img,topk=5,power ='gpu')

print("These are your predicted classes with associated probabilities ")
print("The classes of prediction are :",predicted_classes)
print("The probablities of prediction are :",probabilities_of_predictions)



