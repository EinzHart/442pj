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
from collections import OrderedDict

# directories
data_dir = '../data/'

train_dir = data_dir+'train/'
test_dir = data_dir+'test/'
validation_dir = data_dir+'validation/'
model_file = data_dir+'model/1.model'

use_gpu = True
multiple_gpu = True
load_model = True
quiet = False
make_validation_output = False # generates validation output if True or test output if False
batch_size =175
num_epoches = 20

data_transform = transforms.Compose([
        transforms.ToTensor() #normalizes to 0-1.0 and transposes
        ])

train_dataset = MyCustomDataset(train_dir,  training_flag = True, transforms=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = MyCustomDataset(validation_dir,  training_flag = False, \
                                    transforms=data_transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=20, shuffle=False)

test_dataset = MyCustomDataset(test_dir,  training_flag = False, transforms=data_transform)
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# model
# model = basicNet()
model = Model()

if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
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
        elif multiple_gpu:
            model.load_state_dict(torch.load(model_file))
        else:
            try:
                model.load_state_dict(torch.load(model_file))
            except:
                # original saved file with DataParallel
                state_dict = torch.load(model_file)
                # create new OrderedDict that does not contain `module.`
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model.load_state_dict(new_state_dict)

        print('model loading complete!')
    else:
        print('no previous model detected, start training')
else:
    print('start training new model')

# training
model.train()
model, loss = train(train_dir,train_dataloader,model,optimizer,loss_fn,num_epoches,use_gpu, quiet)
print('Training complete, final loss=',loss.data)
torch.save(model.state_dict(), model_file)
print('Model saved')

# generate output
if make_validation_output:
    # empty folder
    for the_file in os.listdir(validation_dir):
        file_path = os.path.join(validation_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    generate_output(validation_dir,validation_dataloader,model,use_gpu)
    print('Validation output written')
    # evaluate validation outputs
    mae=evaluate(validation_dir+'/normal',validation_dir+'/gt', validation_dir+'/mask')
    print('MAE = ', mae)
else:
    generate_output(test_dir,test_dataloader,model,use_gpu)
    print('Test output written')
