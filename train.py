import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, Function

def train(train_dir,train_dataloader,model,optimizer,criterion,num_epoches,use_gpu):

    for epoch in range(num_epoches):
        print('epoch',epoch)
        running_loss = 0.0
        for i, minibatch in enumerate(train_dataloader):
            # print('minibatch',i)
            if torch.cuda.is_available() and use_gpu:
                inputs = Variable(minibatch['color'].type(torch.FloatTensor).cuda() )
                masks = Variable(minibatch['mask'].type(torch.FloatTensor).cuda() )
                targets = Variable(minibatch['normal'].type(torch.FloatTensor).cuda())
            else:
                inputs = Variable(minibatch['color'].type(torch.FloatTensor) )
                masks = Variable(minibatch['mask'].type(torch.FloatTensor) )
                targets = Variable(minibatch['normal'].type(torch.FloatTensor) )

            optimizer.zero_grad()

            # inputs = inputs * masks
            outputs = model(inputs)
            outputs = outputs * masks
            targets = targets * masks

            # loss = criterion(outputs, targets, masks) #MAE_loss
            loss = criterion(outputs, targets) #MSELoss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            if i % 100 == 99:
                torch.save(model.state_dict(), train_dir+'_temp.model')
        # print('Final loss in epoch ',epoch,':',loss.data)

    return model, loss
