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


class L3Attack(torch.autograd.Function):
    @staticmethod
    def forward(self, model, img, target_lable, datast, allstep, sink_lr, s_radius):
        return L3_function(model, img, target_lable, datast=datast, allstep=allstep, lr=sink_lr, s_radius=s_radius)

    @staticmethod
    def backward(self, grad_output):
        return None, grad_output, None, None, None, None, None


class L4Attack(torch.autograd.Function):
    @staticmethod
    def forward(self, model, img, datast, allstep, sink_lr, s_radius):
        return L4_function(model, img, datast=datast, allstep=allstep, lr=sink_lr, s_radius=s_radius)

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
                datast, 
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
            output = model(transform(x_var, datast=datast))
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
                datast, 
                allstep, 
                lr, 
                u_radius,
                margin=20,
                use_margin=False):
    x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
    true_label = model(transform(x_var.clone(), datast=datast)).data.max(1, keepdim=True)[1][0].item()
    optimizer_s = optim.SGD([x_var], lr=lr)
    with torch.enable_grad():
        for step in range(allstep):
            optimizer_s.zero_grad()
            output = model(transform(x_var, datast=datast))
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
        datast='imagenet', 
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
    true_label = model(transform(x_var.clone(), datast=datast)).data.max(1, keepdim=True)[1][0].item()
    original_softmax = F.softmax(model(transform(x_var.clone(), datast=datast))).data
    optimizer = optim.Adam([x_var], lr=lr)
    target_label = random_label(true_label, datast=datast)
    target_l = torch.LongTensor([target_label]).cuda()
    target_dist = target_distribution(original_softmax, target_label)

    for i in range(allstep):

        optimizer.zero_grad()
        total_loss = 0

        output_ori = model(transform(x_var, datast=datast))
        loss1 = cross_entropy(output_ori, target_dist)  # loss of original image, should descend

        if setting == 'white':
            total_loss += lbd * loss1

            noise_var = noisy(x_var, noise_radius)
            output_noise = model(transform(noise_var, datast=datast))
            loss2 = torch.norm(F.softmax(output_noise) - F.softmax(output_ori),
                               1)  # l1(noisy_img-origin_img), should descend
            total_loss += loss2

            new_target = torch.LongTensor([random_label(target_label, datast=datast)]).cuda()
            t_attack_var = t_attack(model, x_var, new_target, datast, 1, targeted_lr, targeted_radius)  # 1 step t_attack
            output_t_attack = model(transform(t_attack_var, datast=datast))
            loss3 = F.cross_entropy(output_t_attack,
                                    new_target)  # 1 step of targeted attack image, should be new_target, descend
            total_loss += loss3

            u_attack_var = u_attack(model, x_var, datast, 1, untargeted_lr, untargeted_radius)  # 1 step u_attack, if you want to do white box attack for inception, then you will need to change 0.1 to 3 here
            output_u_attack = model(transform(u_attack_var, datast=datast))
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
       datast='imagenet', 
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
    true_label = model(transform(x_var.clone(), datast=datast)).data.max(1, keepdim=True)[1][0].item()
    optimizer = optim.Adam([x_var], lr=lr)
    target_label = random_label(true_label, datast=datast)

    for step in range(allstep):
        optimizer.zero_grad()
        total_loss = 0

        output_ori = model(transform(x_var, datast=datast))
        _, top2_1 = output_ori.data.cpu().topk(2)
        argmax11 = top2_1[0][0]
        if argmax11 == target_label:
            argmax11 = top2_1[0][1]
        loss1 = (output_ori[0][argmax11] - output_ori[0][target_label] + margin).clamp(min=0)

        if setting == 'white':
            total_loss += lbd * loss1  # loss of original image, should descend

            noise_var = noisy(x_var, noise_radius)
            output_noise = model(transform(noise_var, datast=datast))
            loss2 = torch.norm(F.softmax(output_noise) - F.softmax(output_ori),
                               1)  # l1(noisy_img-origin_img), should descend
            total_loss += loss2

            new_tl = random_label(target_label, datast=datast)
            new_target = torch.LongTensor([new_tl]).cuda()
            t_attack_var = t_attack(model, x_var, new_target, datast, 1, targeted_lr, targeted_radius)  # 1 step t_attack
            output_t_attack = model(transform(t_attack_var, datast=datast))
            _, top2_3 = output_t_attack.data.cpu().topk(2)
            argmax13 = top2_3[0][0]
            if argmax13 == new_tl:
                argmax13 = top2_3[0][1]
            loss3 = (output_t_attack[0][argmax13] - output_t_attack[0][new_tl] + margin).clamp(
                min=0)  # 1 step of targeted attack image, should be new_target, descend
            total_loss += loss3  # loss of sink image, should descend

            u_attack_var = u_attack(model, x_var, datast, 1, untargeted_lr, untargeted_radius)  # 1 step u_attack, if you want to do white box attack for inception, then you will need to change 0.1 to 3 here
            output_u_attack = model(transform(u_attack_var, datast=datast))
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
parser.add_argument('--datast', type=str, default='imagenet', help='dataset, imagenet or cifar')
parser.add_argument('--base', type=str, default="resnet")#Note that if you want to generate whitebox adversary for Inception, you may need to change one parameter in attack, untargeted_lr, from 0.1 for resenet to 3 for inception, you can find this explanation in attack.
parser.add_argument('--setting', type=str, default='white', help='attack, white or gray')
parser.add_argument('--allstep', type=int, default=50)
parser.add_argument('--lowbd', type=int, default=0)
parser.add_argument('--upbd', type=int, default=1000)#how many adversaries will be generated
parser.add_argument('--radius', type=float, default=0.1)#how many adversaries will be generated
parser.add_argument('--real_dir', type=str, default='/home/')#this is the folder for real images in ImageNet in .pt format
parser.add_argument('--adv_dir', type=str, default='/home/')#this is the folder to store generate adversaries of ImageNet in .pt format
args = parser.parse_args()

t_attack = L3Attack.apply
u_attack = L4Attack.apply
noisy = Noisy.apply

if args.datast == 'imagenet':
    args.noise_radius = 0.1
    args.targeted_lr = 0.005
    args.targeted_radius = 0.03
    args.untargeted_radius = 0.03
    if args.base == 'resnet':
        model = models.resnet101(pretrained=True)
        args.untargeted_lr = 0.1
    elif args.base == 'inception':
        model = models.inception_v3(pretrained=True, transform_input=False)
        args.untargeted_lr = 3
    else:
        raise Exception('No such model predefined.')
    model = torch.nn.DataParallel(model).cuda()
elif args.datast == 'cifar':
    args.noise_radius = 0.01
    args.targeted_lr = 0.0005
    args.targeted_radius = 0.5
    args.untargeted_radius = 0.5
    args.untargeted_lr = 1
    testset = torchvision.datasets.CIFAR10(root=args.real_dir, train=False, download=True,
                                           transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(), ]))
    if args.base == "vgg":
        model = vgg19()
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load(save_dir + '/model_best.pth.tar')#save directory for vgg19 model
    else:
        raise Exception('No such model predefined.')
    model.load_state_dict(checkpoint['state_dict'])
else:
    raise Exception('Not supported dataset.')


model.eval()
adv_d = args.adv_dir + args.base
t = "_adv0p1_" + str(args.allstep)
numcout = 0
for i in range(args.lowbd, args.upbd):
    if args.datast == 'imagenet':
        view_data = torch.load(args.real_dir + 'real_img/real_' + str(i) + '_img.pt')#load the real images
        view_data_label = torch.load(args.real_dir + 'real_label/real_' + str(i) + '_label.pt')  # real label of this image
    else:
        view_data, view_data_label = testset[i]
        view_data = view_data.unsqueeze(0).cuda()
        view_data_label = view_data_label * torch.ones(1).cuda().long()
    model.eval()
    predicted_label = model(transform(view_data.clone(), datast=args.datast)).data.max(1, keepdim=True)[1][0]
    if predicted_label != view_data_label:
        continue#note that only load images that were classified correctly
    torch.save(view_data_label, adv_d + '/real_label/' + args.datast + '_' + str(numcout) + t + '_label.pt')
    torch.save(PGD(model, view_data, datast=args.datast, allstep=args.allstep, radius=args.radius, setting=args.setting, noise_radius=args.noise_radius, targeted_lr = args.targeted_lr, targeted_radius = args.targeted_radius, untargeted_lr = args.untargeted_lr, untargeted_radius = args.untargeted_radius), adv_d + '/aug_pgd/' + args.datast + '_' + str(numcout) + t + '.pt')
    torch.save(CW(model, view_data, datast=args.datast, allstep=args.allstep, radius=args.radius,  setting=args.setting, noise_radius=args.noise_radius, targeted_lr = args.targeted_lr, targeted_radius = args.targeted_radius, untargeted_lr = args.untargeted_lr, untargeted_radius = args.untargeted_radius), adv_d + '/cw/' + args.datast + '_' + str(numcout) + t + '.pt')
    numcout += 1
print('Finish generating white box adversaries')
