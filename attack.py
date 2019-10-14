import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import argparse
from train_vgg19 import vgg19
from utils import Noisy,transform,random_label 
import os

class L3Attack(torch.autograd.Function):
    @staticmethod
    def forward(self, model, img, target_lable, dataset, allstep, sink_lr, s_radius):
        return L3_function(model, img, target_lable, dataset=dataset, allstep=allstep, lr=sink_lr, s_radius=s_radius)

    @staticmethod
    def backward(self, grad_output):
        return None, grad_output, None, None, None, None, None


class L4Attack(torch.autograd.Function):
    @staticmethod
    def forward(self, model, img, dataset, allstep, sink_lr, u_radius):
        return L4_function(model, img, dataset=dataset, allstep=allstep, lr=sink_lr, u_radius=u_radius)

    @staticmethod
    def backward(self, grad_output):
        return None, grad_output, None, None, None, None


""" 
    Return the variable used for L3 function specified in paper
    [lr] specifies the learning rate of the attack algorithm A
    [s_radius] specifies the maximum l infinity distance between the origianl [img]
"""
def L3_function(model, 
                img, 
                target_lable, 
                dataset, 
                allstep, 
                lr, 
                s_radius,
                margin=20,
                use_margin=False):
    x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
    optimizer_s = optim.SGD([x_var], lr=lr)
    with torch.enable_grad():
        for step in range(allstep):
            optimizer_s.zero_grad()
            output = model(transform(x_var, dataset=dataset))
            if use_margin:
                target_lable = target_lable[0].item()
                _, top2_1 = output.data.cpu().topk(2)
                argmax11 = top2_1[0][0]
                if argmax11 == target_l:
                    argmax11 = top2_1[0][1]
                loss = (output[0][argmax11] - output[0][target_l] + margin).clamp(min=0)
            else:
                loss = F.cross_entropy(output, target_lable)
            loss.backward()
            x_var.data = torch.clamp(x_var - lr * x_var.grad.data, min=0, max=1)
            x_var.data = torch.clamp(x_var - img, min=-s_radius, max=s_radius) + img
    return x_var

""" 
    Return the variable used for L4 function specified in paper
    [lr] specifies the learning rate of the attack algorithm A
    [s_radius] specifies the maximum l infinity distance between the origianl [img]
"""
def L4_function(model, 
                img, 
                dataset, 
                allstep, 
                lr, 
                u_radius,
                margin=20,
                use_margin=False):
    x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
    true_label = model(transform(x_var.clone(), dataset=dataset)).data.max(1, keepdim=True)[1][0].item()
    optimizer_s = optim.SGD([x_var], lr=lr)
    with torch.enable_grad():
        for step in range(allstep):
            optimizer_s.zero_grad()
            output = model(transform(x_var, dataset=dataset))
            if use_margin:
                _, top2_1 = output.data.cpu().topk(2)
                argmax11 = top2_1[0][0]
                if argmax11 == true_label:
                    argmax11 = top2_1[0][1]
                loss = (output[0][true_label] - output[0][argmax11] + margin).clamp(min=0)
            else:
                loss = -F.cross_entropy(output, torch.LongTensor([true_label]).cuda())
            loss.backward()
            x_var.data = torch.clamp(x_var - lr * x_var.grad.data, min=0, max=1)
            x_var.data = torch.clamp(x_var - img, min=-u_radius, max=u_radius) + img
    return x_var

def noisy_img(img, n_radius):
    return img + n_radius * torch.randn_like(img)

""" Return the probability-match cross entropy """
def cross_entropy(pred, target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- target * logsoftmax(pred), dim=1))

def target_distribution(original_softmax, target_label):
    true_label = original_softmax.max(1, keepdim=True)[1][0].item()
    target_l = original_softmax.clone()
    temp = target_l.clone()[0, int(true_label)]
    target_l[0, int(true_label)] = target_l[0, int(target_label)]
    target_l[0, int(target_label)] = temp
    return target_l

""" Return the best effort whitebox or gray box PGD attack
    [allstep] specifies the number of steps
    [lr] specifies the learning rate
    [radius] specifies the maximum l infinity distance between the origianl [img]
    [lbd] specifies the weight we add on L1 loss
    [setting] can be 'white' or 'gray'
"""
def PGD(model, 
        img, 
        dataset='imagenet', 
        allstep=30, 
        lr=0.03, 
        radius=0.1, 
        lbd=2, 
        setting='white', 
        noise_radius=0.1, 
        targeted_lr = 0.005, 
        targeted_radius = 0.03, 
        untargeted_lr = 0.1, 
        untargeted_radius = 0.03):
    model.eval()
    x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
    true_label = model(transform(x_var.clone(), dataset=dataset)).data.max(1, keepdim=True)[1][0].item()
    original_softmax = F.softmax(model(transform(x_var.clone(), dataset=dataset))).data
    optimizer = optim.Adam([x_var], lr=lr)
    target_label = random_label(true_label, dataset=dataset)
    target_l = torch.LongTensor([target_label]).cuda()
    target_dist = target_distribution(original_softmax, target_label)

    for i in range(allstep):

        optimizer.zero_grad()
        total_loss = 0

        output_ori = model(transform(x_var, dataset=dataset))
        loss1 = cross_entropy(output_ori, target_dist)  # loss of original image, should descend

        if setting == 'white':
            total_loss += lbd * loss1

            noise_var = noisy(x_var, noise_radius)
            output_noise = model(transform(noise_var, dataset=dataset))
            loss2 = torch.norm(F.softmax(output_noise) - F.softmax(output_ori),
                               1)  # l1(noisy_img-origin_img), should descend
            total_loss += loss2

            new_target = torch.LongTensor([random_label(target_label, dataset=dataset)]).cuda()
            t_attack_var = t_attack(model, x_var, new_target, dataset, 1, targeted_lr, targeted_radius)  # 1 step t_attack
            output_t_attack = model(transform(t_attack_var, dataset=dataset))
            loss3 = F.cross_entropy(output_t_attack,
                                    new_target)  # 1 step of targeted attack image, should be new_target, descend
            total_loss += loss3

            u_attack_var = u_attack(model, x_var, dataset, 1, untargeted_lr, untargeted_radius)  # 1 step u_attack, if you want to do white box attack for inception, then you will need to change 0.1 to 3 here
            output_u_attack = model(transform(u_attack_var, dataset=dataset))
            loss4 = F.cross_entropy(output_u_attack,
                                    target_l)  # 1 step of u_targeted attack, should be away from target_l, ascend
            total_loss -= loss4

        elif setting == 'gray':
            total_loss += loss1

        else:
            raise "attack setting is not supported"

        total_loss.backward()
        optimizer.step()
        x_var.data = torch.clamp(torch.clamp(x_var, min=0, max=1) - img, min=-radius, max=radius) + img

    return x_var

""" Return the best effort whitebox or gray box CW attack
    [allstep] specifies the number of steps
    [lr] specifies the learning rate
    [radius] specifies the maximum l infinity distance between the origianl [img]
    [lbd] specifies the weight we add on L1 loss
    [setting] can be 'white' or 'gray'
"""
def CW(model, 
       img, 
       dataset='imagenet', 
       allstep=30, 
       lr=0.03, 
       radius=0.1, 
       margin=20.0, 
       lbd=2, 
       setting='white', 
       noise_radius=0.1, 
       targeted_lr = 0.005, 
       targeted_radius = 0.03, 
       untargeted_lr = 0.1, 
       untargeted_radius = 0.03):
    model.eval()
    x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
    true_label = model(transform(x_var.clone(), dataset=dataset)).data.max(1, keepdim=True)[1][0].item()
    optimizer = optim.Adam([x_var], lr=lr)
    target_label = random_label(true_label, dataset=dataset)

    for step in range(allstep):
        optimizer.zero_grad()
        total_loss = 0

        output_ori = model(transform(x_var, dataset=dataset))
        _, top2_1 = output_ori.data.cpu().topk(2)
        argmax11 = top2_1[0][0]
        if argmax11 == target_label:
            argmax11 = top2_1[0][1]
        loss1 = (output_ori[0][argmax11] - output_ori[0][target_label] + margin).clamp(min=0)

        if setting == 'white':
            total_loss += lbd * loss1  # loss of original image, should descend

            noise_var = noisy(x_var, noise_radius)
            output_noise = model(transform(noise_var, dataset=dataset))
            loss2 = torch.norm(F.softmax(output_noise) - F.softmax(output_ori),
                               1)  # l1(noisy_img-origin_img), should descend
            total_loss += loss2

            new_tl = random_label(target_label, dataset=dataset)
            new_target = torch.LongTensor([new_tl]).cuda()
            t_attack_var = t_attack(model, x_var, new_target, dataset, 1, targeted_lr, targeted_radius)  # 1 step t_attack
            output_t_attack = model(transform(t_attack_var, dataset=dataset))
            _, top2_3 = output_t_attack.data.cpu().topk(2)
            argmax13 = top2_3[0][0]
            if argmax13 == new_tl:
                argmax13 = top2_3[0][1]
            loss3 = (output_t_attack[0][argmax13] - output_t_attack[0][new_tl] + margin).clamp(
                min=0)  # 1 step of targeted attack image, should be new_target, descend
            total_loss += loss3  # loss of sink image, should descend

            u_attack_var = u_attack(model, x_var, dataset, 1, untargeted_lr, untargeted_radius)  # 1 step u_attack, if you want to do white box attack for inception, then you will need to change 0.1 to 3 here
            output_u_attack = model(transform(u_attack_var, dataset=dataset))
            _, top2_4 = output_u_attack.data.cpu().topk(2)
            argmax14 = top2_4[0][1]
            if argmax14 == target_label:
                argmax14 = top2_4[0][0]
            loss4 = (output_u_attack[0][argmax14] - output_u_attack[0][target_label] + margin).clamp(
                min=0)  # 1 step of u_targeted attack, should be away from target_l, ascend
            total_loss -= loss4

        elif setting == 'gray':
            total_loss += loss1

        else:
            raise "attack setting is not supported"

        total_loss.backward()
        optimizer.step()
        x_var.data = torch.clamp(torch.clamp(x_var, min=0, max=1) - img, min=-radius, max=radius) + img
    return x_var

parser = argparse.ArgumentParser(description='PyTorch White Box Adversary Generation')
parser.add_argument('--real_dir', type=str, required=True, help='directory to store images correctly classified')
parser.add_argument('--adv_dir', type=str, required=True, help='directory to store adversarial images')
parser.add_argument('--name', type=str, default='_demo_',required=True, help='the name of the adversarial example')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset, imagenet or cifar')
parser.add_argument('--setting', type=str, default='white', help='attack, white or gray')
parser.add_argument('--allstep', type=int, default=50, help='number of steps to run an iterative attack')
parser.add_argument('--base', type=str, default="resnet", help='model, vgg for cifar and resnet/inception for imagenet')
parser.add_argument('--lowbd', type=int, default=0, help='index of the first adversarial example to load')
parser.add_argument('--upbd', type=int, default=1000, help='index of the last adversarial example to load')
parser.add_argument('--radius', type=float, default=0.1, help='adversarial radius')
args = parser.parse_args()

t_attack = L3Attack.apply
u_attack = L4Attack.apply
noisy = Noisy.apply

real_d = os.path.join(args.real_dir,args.base)
adv_d = os.path.join(args.adv_dir,args.base)

if args.dataset == 'imagenet':
    data_dir = './imagenetdata/'
    if not os.path.exists(adv_d):
        os.makedirs(adv_d)
        os.makedirs(os.path.join(adv_d,'pgd'))
        os.makedirs(os.path.join(adv_d,'cw'))
    if not os.path.exists(real_d):
        os.makedirs(real_d)
    noise_radius = 0.1
    targeted_lr = 0.005
    targeted_radius = 0.03
    untargeted_radius = 0.03
    #### use ImageFolder to load images, need to map label correct with target_transform
    testset = torchvision.datasets.ImageFolder(root=data_dir,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),]),
                                               )
    if args.base == 'resnet':
        model = models.resnet101(pretrained=True)
        untargeted_lr = 0.1
    elif args.base == 'inception':
        model = models.inception_v3(pretrained=True, transform_input=False)
        untargeted_lr = 3
    else:
        raise Exception('No such model predefined.')
    model = torch.nn.DataParallel(model).cuda()
elif args.dataset == 'cifar':
    data_dir = './cifardata/'
    if not os.path.exists(adv_d):
        os.makedirs(adv_d)
    if not os.path.exists(real_d):
        os.makedirs(real_d)
    noise_radius = 0.01
    targeted_lr = 0.0005
    targeted_radius = 0.5
    untargeted_radius = 0.5
    untargeted_lr = 1
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                           transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(), ]))

    if args.base == "vgg":
        model = vgg19()
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load('./vgg19model/model_best.pth.tar')#save directory for vgg19 model
    else:
        raise Exception('No such model predefined.')
    model.load_state_dict(checkpoint['state_dict'])
else:
    raise Exception('Not supported dataset.')


model.eval()
title = args.name + str(args.allstep)
numcout = 0
for i in range(args.lowbd, args.upbd):
    view_data, view_data_label = testset[i]
    view_data = view_data.unsqueeze(0).cuda()
    view_data_label = view_data_label * torch.ones(1).cuda().long()
    model.eval()
    predicted_label = model(transform(view_data.clone(), dataset=args.dataset)).data.max(1, keepdim=True)[1][0]
    if predicted_label != view_data_label:
        continue#note that only load images that were classified correctly
    torch.save(view_data, os.path.join(real_d, str(numcout) + '_img.pt'))
    torch.save(view_data_label, os.path.join(real_d, str(numcout) + '_label.pt'))
    torch.save(PGD(model, 
                   view_data, 
                   dataset = args.dataset, 
                   allstep = args.allstep, 
                   radius = args.radius,
                   setting = args.setting, 
                   noise_radius = noise_radius, 
                   targeted_lr = targeted_lr,
                   targeted_radius = targeted_radius, 
                   untargeted_lr = untargeted_lr,
                   untargeted_radius = untargeted_radius),
               os.path.join(os.path.join(adv_d, 'pgd'), str(numcout) + title + '.pt'))
    torch.save(CW(model, 
                  view_data, 
                  dataset = args.dataset, 
                  allstep = args.allstep, 
                  radius = args.radius,
                  setting = args.setting, 
                  noise_radius = noise_radius, 
                  targeted_lr  = targeted_lr,
                  targeted_radius = targeted_radius, 
                  untargeted_lr = untargeted_lr,
                  untargeted_radius = untargeted_radius),
               os.path.join(os.path.join(adv_d, 'cw'), str(numcout) + title + '.pt'))
    numcout += 1
print('Finish generating white box adversaries')
