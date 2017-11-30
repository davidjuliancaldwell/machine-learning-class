########################################################################
# %%

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
sns.set()
sns.set_style("white")
%matplotlib inline
########################################################################

### load in data

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
# %%

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).
# %%

class Net_simple(nn.Module):
    def __init__(self):
        super(Net_simple, self).__init__()
        self.fc1 = nn.Linear(3072, 10)


    def forward(self, x):
        x = x.view(-1,3072)
        x = self.fc1(x)
        return x

class Net_hidden(nn.Module):
    def __init__(self,M):
        super(Net_hidden, self).__init__()
        self.fc1 = nn.Linear(3072, M)
        self.fc2 = nn.Linear(M,10)

    def forward(self, x):
        x = x.view(-1,3072)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_conv(nn.Module):
    def __init__(self,M,p,N):
        super(Net_conv, self).__init__()
        # input, output, square conv are order of inputs for Conv2d
        self.conv1 = nn.Conv2d(3,M,p)
        self.pool = nn.MaxPool2d(N,N)
        self.fc1 = nn.Linear(int((M*((33-p)/N)**2)),int(10))
        self.M = M
        self.p = p
        self.N = N

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1,int((self.M*((33-self.p)/self.N)**2)))
        x = self.fc1(x)
        return x

def run_epochs(trainLoader,testLoader,num_epochs,hyperParameter=True):
    cuda_avail = torch.cuda.is_available()

    iteration_vec = []
    train_accuracy_vec = []
    test_accuracy_vec = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if cuda_avail:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # train classification accuracy
        print('epoch {}'.format(epoch+1))

        correct = 0
        total = 0
        if not(hyperParameter):

            for data in trainloader:
                images, labels = data
                if cuda_avail:
                    images = Variable(images.cuda())
                    labels = labels.cpu()
                else:
                    images = Variable(images)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if cuda_avail:
                    correct += (predicted.cpu() == labels).sum()

                else:
                    correct += (predicted == labels).sum()


            train_accuracy_vec.append(100 * correct / total)
            print('Accuracy of the network on the train images: %d %%' % (
                100 * correct / total))

            # test classification accuracy
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                if cuda_avail:
                    images = Variable(images.cuda())
                    labels = labels.cpu()
                else:
                    images = Variable(images)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if cuda_avail:
                    correct += (predicted.cpu() == labels).sum()

                else:
                    correct += (predicted == labels).sum()

            test_accuracy_vec.append(100 * correct / total)
            print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

        else:
            # validation classification accuracy
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                if cuda_avail:
                    images = Variable(images.cuda())
                    labels = labels.cpu()
                else:
                    images = Variable(images)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if cuda_avail:
                    correct += (predicted.cpu() == labels).sum()

                else:
                    correct += (predicted == labels).sum()

            test_accuracy_vec.append(100 * correct / total)
            print('Accuracy of the network on the validation images: %d %%' % (
                100 * correct / total))

        iteration_vec.append(epoch+1)

    if not(hyperParameter):
        return iteration_vec,train_accuracy_vec,test_accuracy_vec
    else:
        return iteration_vec,test_accuracy_vec


def run_simple_hyper(trainloader,testloader,stepSize_vec,momentum_vec,num_epochs):
    accuracy_vec = np.zeros((stepSize_vec.shape[0],momentum_vec.shape[0]))

    for i,stepSize in enumerate(stepSize_vec):

        for j,momentum in enumerate(momentum_vec):
            net= Net_simple()

            if cuda_avail:
                net.cuda()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=stepSize, momentum=momentum)
            iteration_vec,validation_accuracy_vec = run_epochs(trainloader,testloader,num_epochs,hyperParameter=True)

            accuracy_vec[i][j] = validation_accuracy_vec[-1]
            print('stepSize: {}, momentum: {}'.format(stepSize,momentum))


    return accuracy_vec

def run_hidden_hyper(trainloader,testloader,stepSize_vec,momentum_vec,M_vec,num_epochs):
    accuracy_vec = np.zeros((stepSize_vec.shape[0],momentum_vec.shape[0],M_vec.shape[0]))

    for M in M_vec:
        M = int(M)
        for i,stepSize in enumerate(stepSize_vec):

            for j,momentum in enumerate(momentum_vec):
                net = Net_hidden(M)
                if cuda_avail:
                    net.cuda()
                criterion = nn.CrossEntropyLoss()


                optimizer = optim.SGD(net.parameters(), lr=stepSize, momentum=momentum)
                iteration_vec,validation_accuracy_vec = run_epochs(trainloader,testloader,num_epochs,hyperParameter=True)

                accuracy_vec[i][j] = validation_accuracy_vec[-1]
                print('M: {}, stepSize: {}, momentum: {}'.format(M,stepSize,momentum))

    return accuracy_vec

def run_conv_hyper(trainloader,testloader,stepSize_vec,momentum_vec,M_vec,p_vec,N_vec,num_epochs):
    accuracy_vec = np.zeros((stepSize_vec.shape[0],momentum_vec.shape[0],M_vec.shape[0],p_vec.shape[0],N_vec.shape[0]))

    for M in M_vec:
        M = int(M)

        for p in p_vec:
            p = int(p)

            for N in N_vec:
                N = int(N)
                for i,stepSize in enumerate(stepSize_vec):

                    for j,momentum in enumerate(momentum_vec):
                        net = Net_conv(M,p,N)
                        if cuda_avail:
                            net.cuda()

                        criterion = nn.CrossEntropyLoss()

                        optimizer = optim.SGD(net.parameters(), lr=stepSize, momentum=momentum)
                        iteration_vec,validation_accuracy_vec = run_epochs(trainloader,testloader,num_epochs,hyperParameter=True)

                        accuracy_vec[i][j] = validation_accuracy_vec[-1]
                        print('M: {}, p: {}, N: {},stepSize: {}, momentum: {}'.format(M,p,N,stepSize,momentum))
    return accuracy_vec
########################################################################
# %%
# CROSS VALIDATE!!!
num_epochs = 2
# simple

stepSize_vec = np.array([0.01,0.1])
momentum_vec = np.array([0.1,0.5,0.9])

accuracy_vec_simple = run_simple_hyper(trainloader,testloader,stepSize_vec,momentum_vec,num_epochs)
b_ind_simp = np.unravel_index(accuracy_vec_simple.argmax(), accuracy_vec_simple.shape)
b_step_simp = stepSize_vec[b_ind_simp[0]]
b_momentum_simp = momentum_vec[b_ind_simp[1]]
# %%
# hidden layer
M_vec = np.array([100,200,300])
accuracy_vec_hidden = run_hidden_hyper(trainloader,testloader,stepSize_vec,momentum_vec,M_vec,num_epochs)
b_ind_hid = np.unravel_index(accuracy_vec_hidden.argmax(), accuracy_vec_hidden.shape)
b_M_hid = M_vec[b_ind_hid[0]]
b_step_hid = stepSize_vec[b_ind_hid[1]]
b_momentum_hid = momentum_vec[b_ind_hid[2]]

# %%
# convolution
p_vec = np.array([3,5,7])
N_vec = np.array([10,14,18])
accuracy_vec_conv = run_conv_hyper(trainloader,testloader,stepSize_vec,momentum_vec,M_vec,p_vec,N_vec,num_epochs)
b_ind_conv = np.unravel_index(accuracy_vec_conv.argmax(), accuracy_vec_conv.shape)
b_M_conv = M_vec[b_ind_conv[0]]
b_p_conv = p_vec[b_ind_conv[1]]
b_N_conv = N_vec[b_ind_conv[2]]
b_step_hid = stepSize_vec[b_ind_conv[3]]
b_momentum_hid = momentum_vec[b_ind_conv[4]]

####################################################################

# train and test with the hyperparameters
# %%
net_simple = Net_simple()
if cuda_avail:
    net_simple.cuda()

M_hidden = 200
net_hidden = Net_hidden(M_hidden)
if cuda_avail:
    net_hidden.cuda()

M_conv = 100
p_conv = 5
N_conv = 14
net_conv = Net_conv(M_conv,p_conv,N_conv)
if cuda_avail:
    net_conv.cuda()


net_vec = [net_simple,net_hidden,net_conv]

num_epochs = 3
net_vec = [net_conv]
for net in net_vec:

    print('Beginning network {}'.format(net))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    iteration_vec = []
    train_accuracy_vec = []
    test_accuracy_vec = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if cuda_avail:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            runningStats = True
            if runningStats:
                running_loss += loss.data[0]
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            # train classification accuracy
        print('epoch {}'.format(epoch+1))

        correct = 0
        total = 0
        for data in trainloader:
            images, labels = data
            if cuda_avail:
                images = Variable(images.cuda())
                labels = labels.cpu()
            else:
                images = Variable(images)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if cuda_avail:
                correct += (predicted.cpu() == labels).sum()

            else:
                correct += (predicted == labels).sum()


        train_accuracy_vec.append(100 * correct / total)
        print('Accuracy of the network on the train images: %d %%' % (
            100 * correct / total))

        # test classification accuracy
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            if cuda_avail:
                images = Variable(images.cuda())
                labels = labels.cpu()
            else:
                images = Variable(images)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if cuda_avail:
                correct += (predicted.cpu() == labels).sum()

            else:
                correct += (predicted == labels).sum()

        test_accuracy_vec.append(100 * correct / total)
        iteration_vec.append(epoch+1)


        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    print('Finished with network {}'.format(net))

    plt.figure(dpi=600)
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.plot(iteration_vec,train_accuracy_vec,label='training')
    plt.plot(iteration_vec,test_accuracy_vec,label='testing')
    plt.title('{}'.format(net))
    plt.legend()



########################################################################
