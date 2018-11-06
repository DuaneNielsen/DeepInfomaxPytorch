import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim import Adam
from torchvision.transforms import ToTensor
import statistics as stats
import models

def precision(confusion):
    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / (correct + incorrect)
    return precision


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    num_classes = 10
    fully_supervised = False

    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    ds = CIFAR10(r'c:\data\tv', download=True, transform=ToTensor())
    len_train = len(ds) // 10 * 9
    len_test = len(ds) - len_train
    train, test = random_split(ds, [len_train, len_test])
    train_l = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_l = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)

    if fully_supervised:
        classifier = nn.Sequential(
            models.Encoder(),
            models.Classifier()
        ).to(device)
    else:
        classifier = models.DeepInfoAsLatent('run3', 29).to(device)

    optim = Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(40):

        ll = []
        batch = tqdm(train_l, total=len_train // batch_size)
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            optim.zero_grad()
            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(str(epoch) + ' Train Loss: ' + str(stats.mean(ll)))
            loss.backward()
            optim.step()

        confusion = torch.zeros(num_classes, num_classes)
        batch = tqdm(test_l, total=len_test // batch_size)
        for x, target in batch:
            ll = []
            x = x.to(device)
            target = target.to(device)

            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(str(epoch) + 'Test Loss: ' + str(stats.mean(ll)))

            _, predicted = y.detach().max(1)

            for item in zip(predicted, target):
                confusion[item[0], item[1]] += 1

        precis = precision(confusion)
        print(precis)

        torch.save(classifier, 'c:/data/deepinfomax/models/run4/w_dim' + str(epoch) + '.mdl')
