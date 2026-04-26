from model import train, get_mnist
def main():#train once because it is saved in a file
    trainset, testset, trainloader, testloader, classes = get_mnist()
    model = train(trainloader)
if __name__ == '__main__':
    main()
