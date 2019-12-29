import torch
import torchvision
import torchvision.transforms as transforms


def get_data_loaders(args):
    normalize = transforms.Normalize((.4914, .4822, .4465), (.2470, .2435, .2616))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    kwargs = {'num_workers': 0, 'pin_memory': True}
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                           transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size * 8,
                                             shuffle=False, **kwargs)

    return trainloader, testloader
