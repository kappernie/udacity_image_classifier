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

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

'''
programmer :Appau Ernest Kofi mensah 
reference on code for train.py  ,predict.py  and fsupport.py are from Medium blog, stackoverflow site, pytorch blog  and google 

description:Thismodule contains the supporting functions for both the training and prediction scripts 
the functions here include the 
1-datasetloader and image transform 
2-setup neural network hyperparameter functions 
3-train and save model function
4-load and predict function
5-image processing function 
'''








#########################################################  dataset loader and image transform  #########################################################


def load_data(imageset_path = "./flowers" ):
  

    data_dir = imageset_path = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

 


    trainset_tforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

   

    testset_tforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_tforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


 
    loadtrain_dataset = datasets.ImageFolder(train_dir, transform=trainset_tforms)
    loadvalidation_dataset = datasets.ImageFolder(valid_dir, transform=validation_tforms)
    loadtest_dataset = datasets.ImageFolder(test_dir ,transform = testset_tforms)


    trainset_dataloader  = torch.utils.data.DataLoader(loadtrain_dataset, batch_size=64, shuffle=True)
    validationset_dataloader = torch.utils.data.DataLoader(loadvalidation_dataset, batch_size =32,shuffle = True)
    testset_dataloader = torch.utils.data.DataLoader(loadtest_dataset, batch_size = 20, shuffle = True)



    return trainset_dataloader , validationset_dataloader, testset_dataloader

######################################################### ########################################### #########################################################





















#########################################################  setup hyperparameters of network  #########################################################


def nn_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001,power='gpu'):
    '''
    Arguments: The architecture for the network(alexnet,densenet121,vgg16), the hyperparameters for the network (hidden layer 1 nodes, dropout and learning rate) and whether to use gpu or not
    Returns: The set up model, along with the criterion and the optimizer fo the Training

    '''

    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'vgg16':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
       
        print("Houston we have a problem {} doesnt seem to be a recognised  valid model..\n are you refereing to any of these \n 1-vgg16 \n 2-densenet121,\n or \n alexnet?".format(structure))
        

    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(arch['densenet121'], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))


        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )

        if torch.cuda.is_available():
            model.cuda()

        return model, criterion, optimizer
    
    
    
    
    
    
    
    
    
    
    
    
    

######################################################### ########################################### #########################################################








data_dir = imageset_path = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

 


trainset_tforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

   

testset_tforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

validation_tforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


 
loadtrain_dataset = datasets.ImageFolder(train_dir, transform=trainset_tforms)
loadvalidation_dataset = datasets.ImageFolder(valid_dir, transform=validation_tforms)
loadtest_dataset = datasets.ImageFolder(test_dir ,transform = testset_tforms)
trainset_dataloader  = torch.utils.data.DataLoader(loadtrain_dataset, batch_size=64, shuffle=True)
validationset_dataloader = torch.utils.data.DataLoader(loadvalidation_dataset, batch_size =32,shuffle = True)
testset_dataloader = torch.utils.data.DataLoader(loadtest_dataset, batch_size = 20, shuffle = True)



   


######################################################## ########################################### #########################################################













#########################################################  training and saving model l#########################################################

def train_and_save_network(model, criterion, optimizer, epochs = 12, print_every=20,  trainset_dataloader  = trainset_dataloader , power='gpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, teh dataset, and whether to use a gpu or not
    Returns: Nothing

    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively

    '''
    steps = 0
    running_loss = 0

    print("############training initialized########### ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainset_dataloader ):
            steps += 1
            if torch.cuda.is_available() :
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(validationset_dataloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                      

                validationlost = validationlost / len(validationset_dataloader )
                accuracy = accuracy /len(validationset_dataloader )
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
                
                
                structure ='densenet121',
                hidden_layer1=120,
                dropout= 0.5,
                lr= 0.001,
                

                 
                model.class_to_idx =  train_data.class_to_idx
                model.cpu 
                torch.save({'structure' :structure,
                             'hidden_layer1':hidden_layer1,
                             'dropout':dropout,
                             'lr':lr,
                             'nb_of_epochs':epochs,
                             'state_dict':model.state_dict(),
                             'class_to_idx':model.class_to_idx},
                             'checkpoint.pth')
                
   
               


               
 


    print("#################### training and saving of model is done #######################")
  
######################################################### ########################################### #########################################################
  


    
    
    
    
    
    
    
    
    
    
    
#########################################################  loading and prediction model  #########################################################

def load_and_predict(path='checkpoint.pth',input_img = '/home/workspace/ImageClassifier/flowers/test/23/image_03454.jpg',topk=5,power='gpu'):
  
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    
    
    
    
    model,_,_ = nn_setup(structure='densenet121' , dropout=0.5,hidden_layer1 = 120,lr = 0.001)
 
  

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    
     #flower prediction code
        
    img_torch = image_processing(input_img)
    img_torch =  img_torch.unsqueeze(0)
    
    img_torch = img_torch.type(torch.cuda.FloatTensor)
   
    output = model.forward( img_torch)
    
    
    probabilities = torch.exp(output)
    predicted_probabilities, predicted_indices = probabilities.topk(topk)
    
    
    predicted_probabilities =predicted_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
   
    predicted_indices = predicted_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
   
  
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    predicted_classes = [idx_to_class[index] for index in   predicted_indices]
    
    return predicted_probabilities, predicted_classes
    


   
 


    
    
    
    
    
    
    
    
    
#######################################################image processing ############################################'''
def image_processing(image):
    img_to_be_transformed = Image.open(image)
   
    image_coversion = transforms.Compose([
        
        transforms.Resize(256),
         
        transforms.CenterCrop(224),
        
        transforms.ToTensor(),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      
   
    ])
    image_adjusted= image_coversion(img_to_be_transformed)
    
    return  image_adjusted

    
   
