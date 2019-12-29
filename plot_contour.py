import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
import os
import torch
import numpy as np
from matplotlib.colors import LogNorm


def plot(config):
    """plot contour maps according to loss values"""
    config.out_model_dir = os.path.join(config.out_dir, 'params', config.model_config)
    config.out_loss_dir = os.path.join(config.out_dir, 'loss', config.model_config)
    config.out_fig_dir = os.path.join(config.out_dir, 'fig', config.model_config)
    levels = np.concatenate([np.logspace(-3, -2, num=3), np.arange(0.1, 8, 0.4)])
    path = os.path.join(
        config.out_loss_dir, 'loss*{}*{}*{}'.format(config.xmin, config.xmax, config.size)
    )
    x = y = np.linspace(config.xmin, config.xmax, num=config.size, endpoint=False)
    X, Y = np.meshgrid(x, y)
    Z = torch.load(path)
    Z = np.clip(Z, 0, 1e4)
    plt.contourf(X, Y, Z, levels, alpha=.75, cmap=plt.cm.hot)
    C = plt.contour(X, Y, Z, levels, colors='black', linewidth=.5)
    plt.clabel(C, inline=True, fontsize=10)
    plt.savefig(os.path.join(config.out_fig_dir, 'cont.png'))
    plt.clf()
    plt.imshow(Z, norm=LogNorm(), cmap='hot')
    plt.colorbar(ticks=np.logspace(-3, 3, num=7))
    plt.savefig(os.path.join(config.out_fig_dir, 'heat.png'))


if __name__ == '__main__':
    # process input parameters
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
    plot(args)
