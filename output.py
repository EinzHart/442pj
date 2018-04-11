import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, Function
from torchvision  import datasets, transforms, utils
import cv2
import numpy as np
from network import basicNet
from custom_dataloader import MyCustomDataset

def generate_output(dir,dataloader,model,use_gpu):
    output_dir = dir+'normal/'
    name = 100
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
            img = img.data.numpy().transpose((1,2,0))

            cv2.imwrite(output_dir+'{}.png'.format(name),img)
            name +=1

if __name__ == '__main__':
    use_gpu = False

    data_dir = '../data_debug/'
    validation_dir = data_dir+'validation/'
    model_file = data_dir+'model/1.model'
    model = basicNet()
    model.load_state_dict(torch.load(model_file, map_location = 'cpu'))

    data_transform = transforms.Compose([
            transforms.ToTensor() #applies torchvision's function which normalizes to 0-1.0 and transposes
            ])
    validation_dataset = MyCustomDataset(validation_dir,  training_flag = False, transforms=data_transform)
    validation_dataloader = DataLoader(validation_dataset, batch_size=20, shuffle=False)
    generate_output(validation_dir,validation_dataloader,model,use_gpu)
