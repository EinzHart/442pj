import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, Function
import torch.nn as nn
from torch import optim
from torchvision  import datasets, transforms, utils
import os
import cv2
import numpy as np
from network import basicNet
from custom_dataloader import MyCustomDataset
from train import train
from output import generate_output
from Evaluation_script_ import evaluate, scan_png_files
from loss import MAE_loss
from hourglass import Model

# directories
data_dir = '../data_debug/'

train_dir = data_dir+'train/'
test_dir = data_dir+'test/'
validation_dir = data_dir+'validation/'
model_file = data_dir+'model/1.model'

use_gpu = False
load_model = False
make_validation_output = True # generates validation output if True or test output if False
batch_size =1
num_epoches = 1

data_transform = transforms.Compose([
        transforms.ToTensor() #applies torchvision's function which normalizes to 0-1.0 and transposes
        ])

train_dataset = MyCustomDataset(train_dir,  training_flag = True, transforms=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_dataset = MyCustomDataset(validation_dir,  training_flag = False, transforms=data_transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=20, shuffle=False)

test_dataset = MyCustomDataset(test_dir,  training_flag = False, transforms=data_transform)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# model
model = basicNet()
# model = Model()

# if torch.cuda.device_count() > 1:
#   model = nn.DataParallel(model)
if torch.cuda.is_available() and use_gpu:
    model = model.cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 2e-5)

# criterion
# loss_fn = MAE_loss          #MAE_loss
loss_fn = nn.MSELoss()    #MSELoss

if load_model:
    # load if previous model exists'
    if os.path.isfile(model_file):
        print('loading previous model...')
        if not use_gpu:
            model.load_state_dict(torch.load(model_file, map_location = 'cpu'))
        else:
            model.load_state_dict(torch.load(model_file))
        print('model loading complete!')
    else:
        print('no previous model detected, start training')
else:
    print('start training new model')

# training
model.train()
model, loss = train(train_dir,train_dataloader,model,optimizer,loss_fn,num_epoches,use_gpu)
print('Training complete, final loss=',loss.data)
torch.save(model.state_dict(), model_file)

# generate output
if make_validation_output:
    generate_output(validation_dir,validation_dataloader,model,use_gpu)
    print('Validation output written')
    # evaluate validation outputs
    mae=evaluate(validation_dir+'/normal',validation_dir+'/gt', validation_dir+'/mask')
    print('MAE = ', mae)
else:
    # generate_output(test_dir,test_dataloader,model,use_gpu)
    # print('Test output written')
    pass
