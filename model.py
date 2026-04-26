import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#####Credit to pytorch forum#####
#https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def get_mnist():
    print("importing")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])#grayscale is just 1

    batch_size = 32 #faster

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')
    torch.save(testloader, './mnist.pth',)
    return trainset, testset, trainloader, testloader, classes

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)#greyscale pics
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def train(trainloader):
    import torch.optim as optim
    print("start training")
    net=Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for i, data in enumerate(trainloader, 0):
        if i>2000: break #limiting training
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if i%100==0:
            print("trained, "+str(i))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('Finished Training')
    #save model once trained
    PATH = './mnist_net.pth'
    torch.save(net.state_dict(), PATH)
    return net
def load_model():
    model=Net()
    model.load_state_dict(torch.load('./mnist_net.pth'))
    testset=torch.load('./mnist.pth', weights_only=False)
    testloader = torch.utils.data.DataLoader(testset,batch_size=32, shuffle=False)
    model.eval()#eval mode for model
    return model,testloader
