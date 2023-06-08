# SGLS: Towards Adversarial Robustness via Self-Guided Label Smoothing
![](https://img.shields.io/badge/license-MIT-green)

Official implementation of "Soften to Defend: Towards Adversarial Robustness via Self-Guided Label Smoothing" by Zhuorong Li, Daiwei Yu.

## Abstract

 Adversarial training (AT) is currently one of the most effective ways to obtain the robustness of deep neural networks against adversarial attacks. However, most AT methods suffer from *robust overfitting*, *i.e.*, a significant generalization gap in adversarial robustness between the training and testing curves. In this paper, we first identify a connection between robust overfitting and noisy hard labels, upon which we propose an effective self-guided label smoothing technique for robust learning to weaken the memorization in AT on noisy labels and thus to mitigating robust overfitting. Specifically, it first utilizes the learned probability distributions to soften the over-confident one-hot labels, and then it guides the training process using the consensus among self-distilled models. Empirical results demonstrate that our method can simultaneously boost the standard accuracy and robust performance across multiple benchmark datasets, attack types, and architectures. In addition, we also provide a set of analyses to dive into our method and the importance of soft labels for robust generalization.

![intro](https://github.com/devicerxx/SGLS/blob/master/img/intro.png)

## Dependencies
- Python >= 3.8.0
- Torch >= 1.8.0
- Torchvision >= 0.9.0

Install required dependencies:
```
pip install -r requirements.txt
```


We also evaluate the robustness with [Auto-Attack](https://github.com/fra31/auto-attack). It can be installed via following source code:
```
pip install git+https://github.com/fra31/auto-attack
```

## Quickstart

### Training option and description

The option for the training method is as follows:
- `dataset`: {`cifar10`,`cifar100`}
- `loss`: {`ce`, `at`, `sgls`}

We utilize ResNet-18 as the base model. To change the option, simply modify `arch`:
- `arch` : {`resnet18`, `wrn34`}

### Training scripts

#### Adversarial training
```
# cifar10 for example
bash ./scripts/cifar10/run_at.sh
```

#### Self-Guided Label Smoothing
```
# cifar10 for example
bash ./scripts/cifar10/run_sgls.sh
```

## Comparison
Results of ResNet-18 on CIFAR-10 (**Final** checkpoint):

|  Method | Clean Accuracy(%) | Robust Accuracy(%) | AutoAttack(%)
|:-------:|:-----------------:|:------------------:|:------------------:|
|   AT   |        82.4       |        41.4        | 40.2|
| +SGLS  |        **83.0**      |         **55.9**        | **50.2**|
|   TRADES  |        82.5      |        50.2        | 46.8 |
| +SGLS |        **83.3**      |        **55.4**        | **50.1** |
> 📋