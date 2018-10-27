import torch
import models
from pathlib import Path
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128

# image size 3, 32, 32
# batch size must be an even number
# shuffle must be True
cifar_10_train_dt = CIFAR10(r'c:\data\tv', download=True, transform=ToTensor())
#dev = Subset(cifar_10_train_dt, range(128))
cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=torch.cuda.is_available())

epoch = 9
model_path = Path(r'c:\data\deepinfomax\models\run1\encoder' + str(epoch))

encoder = models.Encoder()
encoder.load_state_dict(torch.load(str(model_path)))
encoder.to(device)

# compute the latent space for each image and store in (latent, image)
minibatches = []
batch = tqdm(cifar_10_train_l, total=len(cifar_10_train_dt) // batch_size)
for images, target in batch:
    images = images.to(device)
    encoded, features = encoder(images)
    i = images.detach().cpu().unbind(0)
    e = encoded.detach().cpu().unbind(0)
    sublist = [elem for elem in zip(e, i)]
    minibatches.append(sublist)

# flatten the minibatches to a single list
ordered = []
for minibatch in minibatches:
    while minibatch:
        ordered.append(minibatch.pop())


def display(subject, ordered):

    def l1_dist(x, y):
        return torch.sum(x - y).item()

    def l2_dist(x, y):
        from math import sqrt
        return sqrt(torch.sum((x - y) ** 2).item())

    # sort by distance to the subject
    ordered = sorted(ordered, key=lambda elem: l2_dist(subject[0], elem[0]))

    subject_repeated = [subject for _ in range(10)]
    nearest_10_images = ordered[:10]
    farthest_10_images = ordered[-10:]

    def make_panel(list_of_images):
        images = [image[1] for image in list_of_images]
        panel = torch.cat(images, dim=2)
        panel_pil = ToPILImage().__call__(panel)
        return panel_pil

    panel_of_subject = make_panel(subject_repeated)
    panel_of_nearest_10 = make_panel(nearest_10_images)
    panel_of_farthest_10 = make_panel(farthest_10_images)

    _img = np.concatenate((panel_of_subject, panel_of_nearest_10, panel_of_farthest_10), axis=0)

    plt.imshow(_img)
    plt.show()


while True:
    # pick a random image
    subject = ordered[random.randrange(0, len(ordered))]
    display(subject, ordered)

