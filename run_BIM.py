from bim_attack import BIM
from model import get_mnist, train
import random
import torch
import matplotlib.pyplot as plt
def show(image,ad,prediction,percent):
    image = image.squeeze().detach().numpy()#need to do this because of matlabplot
    plt.imshow(image,cmap='gray')#grayscale #above before detach did permute(1, 2, 0)
    plt.title(ad)
    plt.xlabel('Prediction: ' +str(prediction)+ ' Percent: ' + str(percent))
    plt.show()

def predict(model,image):
    with torch.no_grad():#not in testing mode anymore
        prediction = model(image)
        prob = torch.softmax(prediction, dim=1)
        pred = prob.argmax(dim=1).item()#prediction
        conf = prob[0,pred].item()#confidence of guess
    return pred, conf
def run_bim1(model,images,labels):
    ##pick image and put into BIM
    for _ in range(10):#10 tests
        index = random.randint(0, len(images) - 1)
        image = images[index].unsqueeze(0)
        label = labels[index].unsqueeze(0)
        og_prediction,percent1 =predict(model,image)

        adversary = BIM(image, model, label)
        ad_prediction,percent2 =predict(model,adversary)
        show(images[index],"OG",og_prediction,percent1)
        show(adversary.squeeze(0), "Adversary", ad_prediction, percent2)
    return 0


def main():
    trainset, testset, trainloader, testloader, classes = get_mnist()
    model=train(trainloader)
    model.eval()#take off training mode
    images,labels=next(iter(testloader))#unpacktestloader
    run_bim1(model,images,labels)

    return 0
if __name__ == "__main__":
    main()
