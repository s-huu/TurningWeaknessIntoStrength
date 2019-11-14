import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import argparse
from detect import l1_vals,targeted_vals,untargeted_vals
from train_vgg19 import vgg19

""" Evaluate the tpr given [fpr] under criteria [opt]"""
def single_metric_fpr_tpr(fpr, 
                          criterions, 
                          model, 
                          dataset, 
                          title, 
                          attacks, 
                          lowind, 
                          upind, 
                          real_dir, 
                          adv_dir, 
                          n_radius,
                          targeted_lr, 
                          t_radius, 
                          untargeted_lr, 
                          u_radius, 
                          opt='l1'):
    if opt == 'l1':
        target = l1_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, n_radius)
        threshold = criterions[fpr][0]
        print('this is l1 norm for real images', target)
    elif opt == 'targeted':
        target = targeted_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, targeted_lr, t_radius)
        threshold = criterions[fpr][1]
        print('this is step of targetd attack for real images', target)
    elif opt == 'untargeted':
        target = untargeted_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir,untargeted_lr, u_radius)
        threshold = criterions[fpr][2]
        print('this is step of untargetd attack for real images', target)
    else:
        raise "Not implemented"

    # Note when opt is "targeted" or "untargeted, the measure is discrete. So we compute a corrected fpr"
    fpr_accurate = len(target[target > threshold]) * 1.0 / len(target)
    print("corresponding accurate fpr of this threshold is ", fpr_accurate)

    for i in range(len(attacks)):
        if opt == 'l1':
            a_target = l1_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, n_radius)
            print('this is l1 norm for ',attacks[i], a_target)
        elif opt == 'targeted':
            a_target = targeted_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir,targeted_lr, t_radius)
            print('this is step of targetd attack for ',attacks[i], a_target)
        elif opt == 'untargeted':
            a_target = untargeted_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, untargeted_lr, u_radius)
            print('this is step of untargetd attack for ',attacks[i], a_target)
        else:
            raise "Not implemented"
        tpr = len(a_target[a_target > threshold]) * 1.0 / len(a_target)
        print("corresponding tpr for " + attacks[i] + "of this threshold is ", tpr)

""" Evaluate the tpr given [fpr] using the combined criteria"""
def combined_metric_fpr_tpr(fpr, 
                            criterions,
                            model, 
                            dataset, 
                            title, 
                            attacks, 
                            lowind, 
                            upind, 
                            real_dir, 
                            adv_dir, 
                            n_radius, 
                            targeted_lr, 
                            t_radius, 
                            untargeted_lr,
                            u_radius):
    target_1 = l1_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, n_radius)
    target_2 = targeted_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, targeted_lr, t_radius)
    target_3 = untargeted_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, untargeted_lr, u_radius)

    fpr_accurate = len(target_1[np.logical_or(np.logical_or(target_1 > criterions[fpr][0], target_2 > criterions[fpr][1]), target_3 > criterions[fpr][2])]) * 1.0 / len(target_1)
    print("corresponding accurate fpr of this threshold is ", fpr_accurate)

    for i in range(len(attacks)):
        a_target_1 = l1_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, n_radius)
        a_target_2 = targeted_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, targeted_lr, t_radius)
        a_target_3 = untargeted_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, untargeted_lr, u_radius)
        tpr = len(a_target_1[np.logical_or(np.logical_or(a_target_1 > criterions[fpr][0], a_target_2 > criterions[fpr][1]),a_target_3 > criterions[fpr][2])]) * 1.0 / len(a_target_1)
        print("corresponding tpr for " + attacks[i] + "of this threshold is ", tpr)

"""Find a set of parameter and corresponding tpr given [target_fpr], you will need to pick a reasonable one, this is 
time consuming, since we need to sort an n^3 number list. What's more, this tuning of three thresholds together is 
different from tune three thresholds individualy, which may leads to same tpr with higher tpr."""
def tune_criterion_thresholds(model,
                              dataset,
                              title,
                              attacks,
                              lowind,
                              upind,
                              real_dir,
                              adv_dir,
                              n_radius,
                              targeted_lr,
                              t_radius,
                              untargeted_lr,
                              u_radius,
                              target_fpr):
    target_1 = l1_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, n_radius)
    target_2 = targeted_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, targeted_lr, t_radius)
    target_3 = untargeted_vals(model, dataset, title, "real", lowind, upind, real_dir, adv_dir, untargeted_lr, u_radius)
    p_1 = target_1.copy()
    p_2 = target_2.copy()
    p_3 = target_3.copy()
    p_1.sort()
    p_2.sort()
    p_3.sort()
    fpr = np.zeros(len(p_1)*len(p_2)*len(p_3)+1)
    for ix_1 in range(0,len(p_1)):
        for ix_2 in range(0,len(p_2)):
            for ix_3 in range(0,len(p_3)):
                fpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] = len(target_1[np.logical_or(np.logical_or(target_1>p_1[ix_1],target_2>p_2[ix_2]),target_3>p_3[ix_3])])*1.0/len(target_1)
    fpr[-1] = len(target_1[np.logical_or(np.logical_or(target_1>=p_1[-1],target_2>=p_2[-1]),target_3>=p_3[-1])])*1.0/len(target_1)
    plt.figure(figsize=(8, 8))
    for i in range(len(attacks)):
        tprs = []
        suitable_pairs = []
        a_target_1 = l1_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, n_radius)[::-1]
        a_target_2 = targeted_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, targeted_lr, t_radius)[::-1]
        a_target_3 = untargeted_vals(model, dataset, title, attacks[i], lowind, upind, real_dir, adv_dir, untargeted_lr, u_radius)[::-1]
        tpr = np.zeros(len(p_1)*len(p_2)+1)
        for ix_1 in range(0,len(p_1)):
            for ix_2 in range(0,len(p_2)):
                for ix_3 in range(0,len(p_3)):
                    tpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] = len(a_target_1[np.logical_or(np.logical_or(a_target_1>p_1[ix_1],a_target_2>p_2[ix_2]),a_target_3>p_3[ix_3])])*1.0/len(a_target_1)
                    if fpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] <= target_fpr and fpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] > target_fpr-0.01:
                        suitable_pairs.append((p_1[ix_1],p_2[ix_2],p_3[ix_3]))
                        tprs.append(tpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3])

        tpr[-1] = len(a_target_1[np.logical_or(np.logical_or(a_target_1>=p_1[-1],a_target_2>=p_2[-1]),a_target_3>=p_3[-1])])*1.0/len(a_target_1)
        return suitable_pairs,tprs
      
parser = argparse.ArgumentParser(description='PyTorch White Box Adversary Detection')
parser.add_argument('--real_dir', type=str, required=True, help='the folder for real images in ImageNet in .pt format')
parser.add_argument('--adv_dir', type=str, required=True, help='the folder to store generate adversaries of ImageNet in .pt')
parser.add_argument('--title', type=str, required=True, help='title of your attack, should be name+step format')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset, imagenet or cifar')
parser.add_argument('--base', type=str, default="resnet", help='model, vgg for cifar and resnet/inception for imagenet')
parser.add_argument('--save_dir', dest='save_dir',help='The directory where the pretrained vgg19 model is saved',default='./vgg19model/', type=str)
parser.add_argument('--lowbd', type=int, default=0, help='index of the first adversarial example to load')
parser.add_argument('--upbd', type=int, default=1000, help='index of the last adversarial example to load')
parser.add_argument('--fpr', type=float, default=0.1, help='false positive rate for detection')
parser.add_argument('--det_opt', type=str, default='combined',help='l1,targeted, untargeted or combined')
args = parser.parse_args()

model = None
if args.dataset == 'imagenet':
    noise_radius = 0.1 #specifies the noise radius in l1 norm detection, used in C1 to measure robustness.
    targeted_lr = 0.005 #specifies the learning rate for targeted attack detection criterion, used in C2t
    targeted_radius = 0.03 #specifies the radius of targeted attack detection criterion, used in C2t
    untargeted_radius = 0.03 #specifies the radius of untargeted attack detection criterion, used in C2u
    if args.base == 'resnet':
        model = models.resnet101(pretrained=True)
        """Criterions on ResNet-101"""
        untargeted_step_threshold = 10000 # C2u threshold, the untargeted attack is most time consuming, you may set this to 1000 or neglect
        ## this detection if no untargeted attack adversary is considered, see our paper for more details.
        criterions = {0.1: (1.90,35,untargeted_step_threshold), 0.2: (1.77, 22, untargeted_step_threshold)}
        untargeted_lr = 0.1 #specifies the learning rate for untargeted attack detection criterion, used in C2u
    elif args.base == 'inception':
        model = models.inception_v3(pretrained=True, transform_input=False)
        """Criterions on Inception"""
        untargeted_step_threshold = 10000 # C2u threshold, the untargeted attack is most time consuming, you may set this to 1000 or neglect
        ## this detection if no untargeted attack adversary is considered, see our paper for more details.
        criterions = {0.1: (1.95, 57, untargeted_step_threshold), 0.2: (1.83, 26, untargeted_step_threshold)}
        untargeted_lr = 3 #specifies the learning rate for untargeted attack detection criterion, used in C2u
    else:
        raise Exception('No such model predefined.')
    model = torch.nn.DataParallel(model).cuda()
elif args.dataset == 'cifar':#need to update parameters in detection like noise_radius, also update criterions
    noise_radius = 0.01 #specifies the noise radius in l1 norm detection, used in C1 to measure robustness.
    targeted_lr = 0.0005 #specifies the learning rate for targeted attack detection criterion, used in C2t
    targeted_radius = 0.5 #specifies the radius of targeted attack detection criterion, used in C2t
    untargeted_radius = 0.5 #specifies the radius of untargeted attack detection criterion, used in C2u
    untargeted_lr = 1 #specifies the learning rate for untargeted attack detection criterion, used in C2u
    if args.base == "vgg":
        model = vgg19()
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load(args.save_dir + '/model_best.pth.tar')#save directory for vgg19 model
        """Criterions on vgg19"""
        untargeted_step_threshold = 10000 # C2u threshold, the untargeted attack is most time consuming, you may set this to 1000 or neglect
        ## this detection if no untargeted attack adversary is considered, see our paper for more details.
        criterions = {0.1: (0.009, 119, untargeted_step_threshold), 0.2: (0.0006, 89, untargeted_step_threshold)}
    else:
        raise Exception('No such model predefined.')
    model.load_state_dict(checkpoint['state_dict'])
else:
    raise Exception('Not supported dataset.')

model.eval()

real_d = os.path.join(args.real_dir,args.base)
adv_d = os.path.join(args.adv_dir,args.base)
attacks = ["pgd", "cw"]
if args.det_opt == 'combined':
    combined_metric_fpr_tpr(args.fpr, 
                            criterions, 
                            model, 
                            args.dataset, 
                            args.title, 
                            attacks, 
                            args.lowbd, 
                            args.upbd, 
                            real_d, 
                            adv_d, 
                            noise_radius, 
                            targeted_lr, 
                            targeted_radius, 
                            untargeted_lr, 
                            untargeted_radius)
else:
    single_metric_fpr_tpr(args.fpr, 
                          criterions, 
                          model, 
                          args.dataset, 
                          args.title, 
                          attacks, 
                          args.lowbd, 
                          args.upbd, 
                          real_d, 
                          adv_d, 
                          noise_radius, 
                          targeted_lr, 
                          targeted_radius, 
                          untargeted_lr, 
                          untargeted_radius, 
                          opt=args.det_opt)

print('finish evaluation based on tuned thresholds')
