import torch.nn as nn
import torch
def BIM(image, model, label):
    epsilon=(10/255) #normilized and changed because of model size
    alpha= (1/255)  #based on (paper) but we normalized by 1/255
    #num_iterations = int(min(4.0 + (epsilon * alpha*255),
    #                1.25 * (epsilon * alpha*255)))  # based on paper
    num_iterations=int(epsilon/alpha)#ten is the perfect number
    adversary= image.clone().detach()

    for i in range(num_iterations):
        adversary.requires_grad= True
        pred= model(adversary)
        loss=nn.CrossEntropyLoss()(pred, label)
        model.zero_grad(set_to_none=True)
        if adversary.grad is not None:
            adversary.grad.zero_()
        loss.backward()
        grad = adversary.grad.sign()
        # steps for interation
        adversary= adversary + alpha * grad
        perturbation =torch.clamp(adversary - image, -epsilon, epsilon)#BIM mixing
        adversary = torch.clamp(perturbation + image, 0,1).detach()
    return adversary
