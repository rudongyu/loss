import os
import random
import math
import argparse
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np


# mean = .1644
# stddev = .3239


def unpickle(file):
    """load raw data from pickle file"""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def prepro_data_args():
    """process config parameters"""
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', default='../cifar10')
    argparser.add_argument('--train_ratio', type=float, default=-1)
    args = argparser.parse_args()
    return args


def load_test_data(config):
    """load test data from given directory"""
    test_file_path = os.path.join(config.data_dir, 'cifar-10-batches-py', 'test_batch')
    dataset = unpickle(test_file_path)
    data, labels = dataset[b'data'], dataset[b'labels']
    data = np.reshape(data, (10000, 3, 32, 32)).astype('float32') / 255.
    return data, labels


def load_train_data(config):
    """load training data with a given ratio"""
    data, labels = [], []
    for fi in range(5):
        train_file_path = os.path.join(config.data_dir, 'cifar-10-batches-py', 'data_batch_{}'.format(fi+1))
        dataset = unpickle(train_file_path)
        data.append(dataset[b'data'])
        labels.append(dataset[b'labels'])
    data, labels = np.concatenate(data), np.concatenate(labels)
    data = np.reshape(data, (50000, 3, 32, 32)).astype('float32') / 255.
    # print(data.mean(axis=(0, 2, 3)), data.std(axis=(0, 2, 3)))
    if config.train_ratio > 0:
        indices = random.sample(range(50000), int(50000*config.train_ratio))
        data, labels = data[indices], labels[indices]
    return data, labels


class CifarDataset(Dataset):
    """process cifar data instances with image reading and normalization"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Normalize((.4914, .4822, .4465), (.2470, .2435, .2616))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image, tgt = torch.FloatTensor(self.data[item]), self.labels[item]
        image = self.transform(image)
        return image, tgt


if __name__ == '__main__':
    batch_size = 64
    config = prepro_data_args()
    train_data = CifarDataset(*load_train_data(config))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    mean = np.zeros(3)
    num = 0
    for batch in tqdm(loader, ncols=50):
        images, targets = batch
        mean += torch.sum(images, dim=(0, 2, 3)).numpy()
        num += len(images)
        print(images.type())
    print(num, mean/(num * 32 * 32))
