import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim import Adam
from torchvision.transforms import ToTensor
import statistics as stats
import models
from pathlib import Path


def precision(confusion):
    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / (correct + incorrect)
    total_correct = correct.sum().item()
    total_incorrect = incorrect.sum().item()
    percent_correct = total_correct / (total_correct + total_incorrect)
    return precision, percent_correct


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    num_classes = 10
    fully_supervised = False
    reload = 169
    run_id = 6
    epochs = 100

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
        classifier = models.DeepInfoAsLatent('run5', '990').to(device)
        if reload is not None:
            classifier = torch.load(f'c:/data/deepinfomax/models/run{run_id}/w_dim{reload}.mdl')

    optim = Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(reload + 1, reload + epochs):

        ll = []
        batch = tqdm(train_l, total=len_train // batch_size)
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            optim.zero_grad()
            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(f'{epoch} Train Loss: {stats.mean(ll)}')
            loss.backward()
            optim.step()

        confusion = torch.zeros(num_classes, num_classes)
        batch = tqdm(test_l, total=len_test // batch_size)
        ll = []
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(f'{epoch} Test Loss: {stats.mean(ll)}')

            _, predicted = y.detach().max(1)

            for item in zip(predicted, target):
                confusion[item[0], item[1]] += 1

        precis = precision(confusion)
        print(precis)

        classifier_save_path = Path('c:/data/deepinfomax/models/run' + str(run_id) + '/w_dim' + str(epoch) + '.mdl')
        classifier_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(classifier, str(classifier_save_path))
