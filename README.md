## A New Defense Against Adversarial Images: Turning a Weakness into a Strength
<img align="right" src="detect_fig.png" width="300px" />

#### Authors:
* [Tao Yu](http://www.cs.cornell.edu/~tyu/)*
* [Shengyuan Hu](https://s-huu.github.io)*
* [Chuan Guo](https://sites.google.com/view/chuanguo)
* [Weilun Chao](http://www-scf.usc.edu/~weilunc/)
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)

*: Equal Contribution

### Introduction
This repo contains official code and models for the NeurIPS 2019 paper, A New Defense Against Adversarial Images: Turning a Weakness into a Strength.

We postulate that if an image has been tampered with adversarial perturbation, then surrouding adversarial directions either become harder to find with gradient methods or have substantially higher density than for natural images. Based on this, we develop a practical test for this signature characteristic to successfully detect both gray-box and white-box adversarial attacks.

A table about experimental results on cifar and imagenet.

### Dependencies
* Python 3
* PyTorch >= 1.0.0
* numpy

### Data
- For CIFAR10 dataset, it is downloaded automatically by torchvision.datasets.CIFAR10() function in train_vgg19.py.
- For ImageNet dataset, download the validation images to a folder `~/imagenetdata/`, then move them into labeled subfolders with
imagenet.sh.

### Usage
Introduction of different files,
```
$ python train_vgg19.py # train vgg19 model on cifar
$ python attack.py # attack to generate adversaries
$ python evaluate.py # detect and evaluate
```

### Citation
If you use our code or wish to refer to our results, please use the following BibTex entry:
```
@InProceedings{Yu_2019_NIPS,
  author = {Yu, Tao and Hu, Shengyuan and Guo, Chuan and Chao, Weilun and Weinberger, Kilian},
  title = {A New Defense Against Adversarial Images: Turning a Weakness into a Strength},
  booktitle = {Proceedings of the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)},
  month = {Oct.},
  year = {2019}
}
```
