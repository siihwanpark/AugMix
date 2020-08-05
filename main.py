import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms

from WideResNet_pytorch.wideresnet import WideResNet

PATH = "./ckpt/wrn40-2.ckpt"

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
    js_loss = False
    batch_size = 256
    # 1. dataload
    # basic augmentation & preprocessing
    train_base_aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ]
    preprocess = [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ]
    train_transform = transforms.Compose(train_base_aug + preprocess)
    test_transform = transforms.Compose(preprocess)
    # load data
    train_data = datasets.CIFAR100('./data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100('./data/cifar', train=False, transform=test_transform, download=True)
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

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    # training model with cifar100
    model.train()
    losses = []
    for epoch in range(epochs):
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.cuda(), targets.cuda()
            optimizer.zero_grad()
            if js_loss:
                pass
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            if i % 100 == 0 or i+1 == len(train_loader):
                print("Train Loss: {:.4f}".format(loss.item()))

        torch.save({
            "epoch": epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses
        }, PATH)
    # evaluate on cifar100-c
    for corruption in CORRUPTIONS:
        test_data.data = np.load('./data/cifar/CIFAR-100-C/%s.npy' % corruption)
        test_data.targets = torch.LongTensor(np.load('./data/cifar/CIFAR-100-C/labels.npy'))

if __name__=="__main__":
    main()
