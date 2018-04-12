import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, Function
from torchvision  import datasets, transforms, utils
import cv2
import numpy as np
from network import basicNet
from custom_dataloader import MyCustomDataset
from hourglass import Model
from collections import OrderedDict

def generate_output(dir,dataloader,model,use_gpu):
    output_dir = dir+'normal/'
    name = 0 #change this to the first image number to be generated from color
    model.eval()

    for i, minibatch in enumerate(dataloader):
        print('minibatch',i)
        if torch.cuda.is_available() and use_gpu:
            inputs = Variable(minibatch['color'].type(torch.FloatTensor).cuda(), requires_grad=False)
            masks = Variable(minibatch['mask'].type(torch.FloatTensor).cuda(), requires_grad=False )
        else:
            inputs = Variable(minibatch['color'].type(torch.FloatTensor), requires_grad=False )
            masks = Variable(minibatch['mask'].type(torch.FloatTensor), requires_grad=False )

        # inputs = inputs * masks
        outputs = model(inputs)
        outputs = ((outputs / 2) + 0.5 )*255
        outputs = torch.clamp(outputs,min=0,max=255)
        outputs = outputs * masks

        n,c,h,w = outputs.size()
        for idx in range(n):
            img = outputs[idx,:,:,:]
            img = img.data.cpu().numpy().transpose((1,2,0))

            cv2.imwrite(output_dir+'{}.png'.format(name),img)
            name +=1

if __name__ == '__main__':
    use_gpu = True

    data_dir = '../data/'
    # output_dir = data_dir+'validation/'
    output_dir = data_dir+'test/'
    model_file = data_dir+'model/1.model'
    # model = basicNet()
    model = Model()
    if not use_gpu:
        model.load_state_dict(torch.load(model_file, map_location = 'cpu'))
    else:
        model = model.cuda()
        model.load_state_dict(torch.load(model_file))

    data_transform = transforms.Compose([
            transforms.ToTensor() #applies torchvision's function which normalizes to 0-1.0 and transposes
            ])
    output_dataset = MyCustomDataset(output_dir,  training_flag = False, transforms=data_transform)
    output_dataloader = DataLoader(output_dataset, batch_size=50, shuffle=False)
    generate_output(output_dir,output_dataloader,model,use_gpu)
