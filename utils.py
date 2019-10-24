import torch
import numpy as np


"""Normalize the data given the dataset. Only ImageNet and CIFAR-10 are supported"""
def transform(img, dataset='imagenet'):
    # Data
    if dataset == 'imagenet':
        mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img).cuda()
    elif dataset == 'cifar':
        mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img).cuda()
    else:
        raise "dataset is not supported"
    return (img - mean) / std


"""Given [label] and [dataset], return a random label different from [label]"""
def random_label(label, dataset='imagenet'):
    if dataset == 'imagenet':
        class_num = 1000
    elif dataset == 'cifar':
        class_num = 10
    else:
        raise "dataset is not supported"
    attack_label = np.random.randint(class_num)
    while label == attack_label:
        attack_label = np.random.randint(class_num)
    return attack_label

"""Given the variance of zero_mean Gaussian [n_radius], return a noisy version of [img]"""
def noisy_img(img, n_radius):
    return img + n_radius * torch.randn_like(img)

class Noisy(torch.autograd.Function):
    @staticmethod
    def forward(self, img, n_radius):
        return noisy_img(img, n_radius=n_radius)

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None
