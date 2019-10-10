## A New Defense Against Adversarial Images: Turning a Weakness into a Strength
<img align="right" src="detect_fig.png" width="300px" />

#### Authors:
* [Tao Yu](http://www.cs.cornell.edu/~tyu/)*
* Shengyuan Hu*
* [Chuan Guo](https://sites.google.com/view/chuanguo)
* [Weilun Chao](http://www-scf.usc.edu/~weilunc/)
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)

*: Equal Contribution

### Introduction
This repo contains official code and models for the NeurIPS 2019 paper, A New Defense Against Adversarial Images: Turning a Weakness into a Strength.

A table about experimental results on cifar and imagenet.

### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Data
How to get cifar and imagenet data

### Usage
Introduction of different files
```
$ python train_vgg19.py # train vgg19 model on cifar
$ python attack.py # attack to generate adversaries
$ python evaluate.py # detect and evaluate
```

### Citation
If you use our code or wish to refer to our results, please use the following BibTex entry:
```
@InProceedings{Yu_2019_NIPS,
  author = {Tao Yu, Shengyuan Hu, Chuan Guo, Weilun Chao and Kilian Q. Weinberger},
  title = {A New Defense Against Adversarial Images: Turning a Weakness into a Strength},
  booktitle = {The 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)},
  month = {Oct.},
  year = {2019}
}
```