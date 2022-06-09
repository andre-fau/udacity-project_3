#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import argparse

###
import json
import logging
import os
import sys
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

###
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

def test(model, test_loader, loss_criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    pass

#def train(model, train_loader, criterion, optimizer):
def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    hook = get_hook(create_if_not_exists=True)
    epoch_times = []
    if hook:
        hook.register_loss(criterion)
        
    # train the model
    for i in range(epochs):
        print("START TRAINING")
        if hook:
            hook.set_mode(modes.TRAIN)
        start = time.time()
        model.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print("START VALIDATING")
        if hook:
            hook.set_mode(modes.EVAL)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print(
            "Epoch %d: train loss %.3f, val loss %.3f, in %.1f sec"
            % (i, train_loss, val_loss, epoch_time)
        )

    # calculate training time after all epoch
    p50 = np.percentile(epoch_times, 50)
    print("Median training time per Epoch=%.1f sec" % p50)
    return model   
    
def net(pre_model):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.__dict__[pre_model](pretrained=True) # default resnet50
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
    nn.Linear(num_features, 133))
    return model

def model_fn(model_dir):
    model = net("resnet50")
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    return torch.utils.data.DataLoader(data, batch_size = batch_size)
    
def main(args):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(args.model)
    model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #loss_criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    ###
    # got input from: https://medium.com/@uijaz59/dog-breed-classification-using-pytorch-207cf27c2031
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    # Image Transformation
    data_transforms = {
        'train' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), # randomly flip and rotate
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),

        'valid' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),

        'test' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }
    # Reading Dataset
    image_datasets = {
        'train' : torchvision.datasets.ImageFolder(root=train_dir,transform=data_transforms['train']),
        'valid' : torchvision.datasets.ImageFolder(root=valid_dir,transform=data_transforms['valid']),
        'test' : torchvision.datasets.ImageFolder(root=test_dir,transform=data_transforms['test'])
    }
    
    train_loader = create_data_loaders(image_datasets['train'], args.batch_size)
    val_loader = create_data_loaders(image_datasets['valid'], args.batch_size)
    test_loader = create_data_loaders(image_datasets['test'], args.batch_size)
    ###
    
    model=train(model, train_loader, val_loader, loss_criterion, optimizer, args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    #test(model, test_loader, criterion)
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    #torch.save(model, args.model_dir)
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="resnet50")
    
    #parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    #parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    #parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
