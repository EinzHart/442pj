import numpy as np
from torch.autograd import Variable, Function
import torch

def MAE_loss(predictions, groundtruths, masks):
    """
    Takes Variables with torch tensors of images (pred, gt, mask) and calculates Mean Average Error

    """

    n,c,h,w = predictions.size()
    assert predictions.size() == groundtruths.size()

    # Measure: mean angle error over all pixels
    mean_angle_error = 0
    total_pixels = 0
    for idx in range(n):
        prediction = predictions[idx,:,:,:]
        groundtruth = groundtruths[idx,:,:,:]
        mask = masks[idx,0,:,:]

        # prediction = ((prediction / 255.0) - 0.5) * 2
        # groundtruth = ((groundtruth / 255.0) - 0.5) * 2

        temp,_=torch.nonzero(mask).size()
        total_pixels += temp
        mask = (mask != 0)
        mask=mask.type(torch.FloatTensor)

        a11 = torch.sum(prediction * prediction, dim=0)
        a22 = torch.sum(groundtruth * groundtruth, dim=0)
        a12 = torch.sum(prediction * groundtruth, dim=0)

        cos_dist = torch.div(a12, torch.sqrt(a11 * a22))
        cos_dist[(cos_dist!=cos_dist).detach()] = -1

        cos_dist =cos_dist * mask
        cos_dist = torch.clamp(cos_dist,min=-1,max=1)
        # cos_dist = torch.abs(cos_dist)
        angle_error = torch.acos(cos_dist)  * mask
        mean_angle_error += torch.sum(angle_error)
        # print('max mask',torch.max(mask))
        # print('min angle',torch.min(angle_error))
        # print('angle error',(angle_error))
        # print('pixels', total_pixels)

    print(mean_angle_error / total_pixels)
    
    return mean_angle_error / total_pixels
