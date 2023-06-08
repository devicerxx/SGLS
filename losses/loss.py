import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class CrossEntropy():
    def __init__(self):
        self.crit = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, index, epoch):
        loss = self.crit(logits, targets)
        return loss


class AT():
    def __init__(self, step_size=2.0 / 255.0, epsilon=8.0 / 255.0, perturb_steps=10, random=True):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random = random

    def __call__(self, x_natural, y, index, epoch, model, optimizer):
        model.eval()
        # generate adversarial example
        if self.random:
            x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = torch.zeros_like(x_natural)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # compute loss
        model.train()
        optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        # calculate robust loss
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        return logits, loss

class SGLS():
    def __init__(self, labels, num_classes=10, momentum=0.9, es=90, step_size=2.0/255.0, epsilon=8.0/255.0, perturb_steps=10,
                    random=True):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random = random

    def __call__(self, x_natural, y, index, epoch, model, optimizer):
        model.eval()
        # generate adversarial example
        if self.random:
            x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = torch.zeros_like(x_natural)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # compute loss
        model.train()
        optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        # calculate robust loss
        logits = model(x_adv)
        # print(y.shape)
        if epoch < self.es:
            loss = F.cross_entropy(logits, y)
        else:
            prob = F.softmax(logits.detach(), dim=1)
            clean_out = model(x_natural)
            clean_prob = F.softmax(clean_out.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] + \
                                      (1 - self.momentum) * (0.2 * clean_prob + 0.8 * prob)
            loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)
            loss = loss.mean()
        return logits, loss
