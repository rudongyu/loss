import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
from models.vgg import VGG
from models.resnet import ResNet
from models.densenet import DenseNet
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from data import CifarDataset, load_train_data, load_test_data
from dataloader import get_data_loaders
import time
import os
import random
import argparse


#  load models according to name
models = {
    'vgg': VGG,
    'resnet': ResNet,
    'densenet': DenseNet
}


def init_params(net):
    """initialize the parameters of CNNs"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def loss_surface_2d(config, model, data_loader):
    """calculate loss values in 2-d hyperplane"""
    delta, eta = OrderedDict(), OrderedDict()
    for k, v in model.state_dict().items():
        # batch normalization or bias parameters are omitted
        if v.dim() <= 1:
            delta[k], eta[k] = torch.zeros_like(v), torch.zeros_like(v)
        # perform filter wise normalization
        else:
            delta[k] = torch.randn_like(v)
            eta[k] = torch.randn_like(v)
            for d1, d2, w in zip(delta[k], eta[k], v):
                d1.mul_(w.norm() / (d1.norm() + 1e-10))
                d2.mul_(w.norm() / (d2.norm() + 1e-10))
    # define axes of contour maps
    alphas = betas = np.linspace(config.xmin, config.xmax, num=config.size, endpoint=False)
    Alpha, Beta = np.meshgrid(alphas, betas)
    loss_values = np.zeros([config.size, config.size])
    perturbed_model = deepcopy(model)
    # calculate the loss value in each grid point
    for i, (alpha, beta) in tqdm(enumerate(zip(Alpha.reshape(-1), Beta.reshape(-1))), ncols=50, total=config.size**2):
        perturbed_state_dict = get_shifted_state_dict(model, alpha, beta, delta, eta)
        loss = get_single_point_loss(perturbed_model, data_loader, perturbed_state_dict)
        loss_values[i // config.size, i % config.size] = loss
    torch.save(loss_values, os.path.join(config.out_loss_dir, 'loss*{}*{}*{}'.format(config.xmin, config.xmax, config.size)))


@torch.no_grad()
def get_single_point_loss(shifted_model, data_loader, perturbed_state_dict):
    """calculate the loss value in a single grid point"""
    shifted_model.load_state_dict(perturbed_state_dict)
    shifted_model.eval()
    total = total_loss = 0
    for inputs, labels in data_loader:
        loss = shifted_model(inputs.to(0), labels.to(0))
        total_loss += loss.item() * len(inputs)
        total += len(inputs)
    return total_loss / total


def get_shifted_state_dict(model, alpha, beta, delta, eta):
    """calculate the shifted parameters in state_dicts"""
    theta = deepcopy(model.state_dict())
    for name in delta:
        # ignore batch normalization or bias parameters
        if delta[name].dim() <= 1:
            continue
        theta[name] += alpha * delta[name] + beta * eta[name]

    return theta


def main(config):
    # initialize random seeds
    random.seed(config.rand_seed)
    np.random.seed(config.rand_seed)
    torch.manual_seed(config.rand_seed)
    torch.cuda.manual_seed_all(config.rand_seed)
    cudnn.benchmark = True

    # initialize output directories
    config.out_model_dir = os.path.join(config.out_dir, 'params', config.model_config)
    config.out_loss_dir = os.path.join(config.out_dir, 'loss', config.model_config)
    config.out_fig_dir = os.path.join(config.out_dir, 'fig', config.model_config)
    for d in [config.out_model_dir, config.out_loss_dir, config.out_fig_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # initialize the model
    model = models[config.model](config.model_config).to(0)
    init_params(model)
    train_loader, test_loader = get_data_loaders(config)

    # train the model from scratch
    if config.train:
        if config.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay,
                                  nesterov=True)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        for epoch in tqdm(range(config.epoch_num), ncols=50):
            running_loss = .0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                loss = model(inputs.to(0), labels.to(0))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(inputs)
            if epoch == 149 or epoch == 224 or epoch == 274:
                scheduler.step()
            print("Epoch {} Average Loss: {}".format(epoch + 1, running_loss / len(train_loader.dataset)))

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images.to(0))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(0)).sum().item()

        # record the model parameters and test acc
        print('Accuracy of the network on the 10000 test images: {}'.format(correct / total * 100))
        state = {'state_dict': model.state_dict(), 'acc': correct/total}
        torch.save(state, os.path.join(config.out_model_dir, 'model_state'))

    # if trained already, load parameters from given directory
    else:
        model.load_state_dict(torch.load(os.path.join(config.out_model_dir, 'model_state'))['state_dict'])
        model.eval()

    # calculate loss values
    loss_surface_2d(config, model, test_loader)


if __name__ == '__main__':
    # set config parameters
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', default='../cifar10')
    argparser.add_argument('--train_ratio', type=float, default=-1)
    argparser.add_argument('--epoch_num', type=int, default=300)
    argparser.add_argument('--out_dir', default='plotout')
    argparser.add_argument('--model', default='vgg')
    argparser.add_argument('--model_config', default='VGG16')
    argparser.add_argument('--optimizer', default='SGD')
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--lr', type=float, default=0.1)
    argparser.add_argument('--lr_decay', type=float, default=0.1)
    argparser.add_argument('--weight_decay', default=0.0005, type=float)
    argparser.add_argument('--rand_seed', default=0, type=int)
    argparser.add_argument('--plot_acc', action="store_true")
    argparser.add_argument('--xmin', type=float, default=-1.0)
    argparser.add_argument('--xmax', type=float, default=1.0)
    argparser.add_argument('--size', type=int, default=50)
    argparser.add_argument('--train', action='store_true')
    args = argparser.parse_args()
    main(args)
