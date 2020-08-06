import time
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from WideResNet_pytorch.wideresnet import WideResNet

from augment_and_mix import AugMixDataset

PATH = "./ckpt/AugMix_epoch_"

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def main():
    torch.manual_seed(2020)
    np.random.seed(2020)
    epochs = 100
    k = 3
    alpha = 1.
    js_loss = True
    batch_size = 256

    # 1. dataload
    # basic augmentation & preprocessing
    train_base_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ])

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    train_transform = train_base_aug
    test_transform = preprocess

    # load data
    train_data = datasets.CIFAR100('./data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100('./data/cifar', train=False, transform=test_transform, download=True)

    train_data = AugMixDataset(train_data, preprocess, k, alpha, not(js_loss))
    train_loader = torch.utils.data.DataLoader(
                   train_data,
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=True)

    # 2. model
    # wideresnet 40-2
    model = WideResNet(depth=40, num_classes=100, widen_factor=2, drop_rate=0.0)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.SGD(
                  model.parameters(),
                  0.1,
                  momentum=0.9,
                  weight_decay=0.0005,
                  nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=1e-6, last_epoch=-1)

    model = nn.DataParallel(model).to(device)
    cudnn.benchmark = True

    # training model with cifar100
    model.train()
    losses = []
    t = time.time()

    for epoch in range(epochs):
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            if js_loss:
                bs = images[0].size(0)
                images_cat = torch.cat(images, dim = 0).to(device) # [3 * batch, 3, 32, 32]
                targets = targets.to(device)

                logits = model(images_cat)
                logits_orig, logits_augmix1, logits_augmix2 = logits[:bs], logits[bs:2*bs], logits[2*bs:]

                loss = F.cross_entropy(logits_orig, targets)

                p_orig, p_augmix1, p_augmix2 = F.softmax(logits_orig, dim = -1), F.softmax(logits_augmix1, dim = -1), F.softmax(logits_augmix2, dim = -1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_orig + p_augmix1 + p_augmix2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
                                F.kl_div(p_mixture, p_augmix1, reduction='batchmean') +
                                F.kl_div(p_mixture, p_augmix2, reduction='batchmean')) / 3.

            else:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            if (i+1) % 10 == 0 or i+1 == len(train_loader):
                print("[%d/%d][%d/%d] Train Loss: %.4f | time : %.2fs"
                        %(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), time.time() - t))
                t = time.time()

        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            torch.save({
                "epoch": epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, PATH+"%d.pt"%(epoch + 1))

    fig, ax = plt.subplots()
    ax.plot(losses, label = 'train loss')
    ax.set_xlabel('iterations')
    ax.set_ylabel('cross entropy loss')
    ax.legend()

    ax.set(title="Loss Curve : AugMix")
    ax.grid()

    fig.savefig("results/AugMix_loss_curve.png")
    plt.close()

    model.eval()
    with torch.no_grad():
        # evaluate on cifar100
        test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True)

        error, total = 0, 0
        print("Test on CIFAR-100")

        t = time.time()
        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            preds = torch.argmax(model(images), dim = -1)
            error += (preds != targets).sum().item()
            total += targets.size(0)

        print("Test error rate on CIFAR-100 : %.4f | time : %.2fs"%((error/total), time.time() - t))

        # evaluate on cifar100-c
        for corruption in CORRUPTIONS:
            print("Test on " + corruption)
            test_data.data = np.load('./data/cifar/CIFAR-100-C/%s.npy' % corruption)
            test_data.targets = torch.LongTensor(np.load('./data/cifar/CIFAR-100-C/labels.npy'))
            test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True)

            error, total = 0, 0

            t = time.time()
            for i, (images, targets) in enumerate(test_loader):
                images, targets = images.to(device), targets.to(device)
                preds = torch.argmax(model(images), dim = -1)
                error += (preds != targets).sum().item()
                total += targets.size(0)

            print("Test error rate on CIFAR-100-C with " + corruption + " : %.4f | time : %.2fs"%(error/total, time.time() - t))

if __name__=="__main__":
    main()
