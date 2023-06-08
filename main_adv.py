import argparse
import os
import shutil
import time
import copy
import PIL.Image as Image
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CIFAR10, CIFAR100
from losses import get_loss
from models import get_model
from utils import get_scheduler, get_optimizer, accuracy, save_checkpoint, AverageMeter

import torchattacks

parser = argparse.ArgumentParser(description='Self-Guided Label Smoothing')
# network
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# training setting
parser.add_argument('--data-root', help='The directory of data',
                    default='~/datasets/CIFAR10', type=str)
parser.add_argument('--dataset', help='dataset used to training',
                    default='cifar10', type=str)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer for training')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-schedule', default='step', type=str,
                    help='LR decay schedule')
parser.add_argument('--lr-milestones', type=int, nargs='+', default=[100, 150],
                    help='LR decay milestones for step schedule.')
parser.add_argument('--lr-gamma', default=0.1, type=float,
                    help='LR decay gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# loss function
parser.add_argument('--loss', default='sgls', help='loss function')
parser.add_argument('--sgls-alpha', default=0.9, type=float,
                    help='momentum term of self-adaptive training')
parser.add_argument('--sgls-es', default=0, type=int,
                    help='start epoch of self-adaptive training (default 0)')
# adv training
parser.add_argument('--epsilon', default=8.0/255.0, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2.0/255.0, type=float,
                    help='perturb step size')
# misc
parser.add_argument('-s', '--seed', default=None, type=int,
                    help='number of data loading workers (default: None)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=150, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-freq', default=1, type=int,
                    help='print frequency (default: 1)')
args = parser.parse_args()


best_prec1 = 0
if args.seed is None:
    import random
    args.seed = random.randint(1, 10000)


def main():
    print(args)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    global best_prec1

    # prepare dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == 'cifar10':
        trainset = CIFAR10(root='~/data/cifar10', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
        num_classes = trainset.num_classes
        targets = np.asarray(trainset.targets)
        testset = CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=args.batch_size*4, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    else:
        trainset = CIFAR100(root='~/data/cifar100', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
        num_classes = trainset.num_classes
        targets = np.asarray(trainset.targets)
        testset = CIFAR100(root='~/data/cifar100', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=args.batch_size*4, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)


    model = get_model(args, num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()

    criterion = get_loss(args, labels=targets, num_classes=num_classes)
    optimizer = get_optimizer(model, args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = get_scheduler(optimizer, args)

    if args.evaluate:
        validate(test_loader, model)
        return

    print("*" * 40)
    for epoch in range(args.start_epoch, args.epochs + 1):
        scheduler.step(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        print("*" * 40)

        # evaluate on validation sets
        prec1 = validate_clean(test_loader, model)
        _, robust_acc = eval_test_pgd(model, test_loader)
        print(f"sa test: {prec1: .2f}, \t , ra test: {robust_acc: .2f}")
        print("*" * 40)

        # remember best prec@1 and save checkpoint
        is_best = robust_acc > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if (epoch < 70 and epoch % 10 == 0) or (epoch >= 70 and epoch % args.save_freq == 0):
            filename = 'checkpoint_{}.tar'.format(epoch)
        else:
            filename = None
        save_checkpoint(args.save_dir, {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=filename)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        # compute output
        output, loss = criterion(input, target, index, epoch, model, optimizer)

        # compute gradient and do SGD step
        loss.backward(retain_graph=True)
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR {lr:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i+1, len(train_loader), lr=lr, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = F.cross_entropy(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def validate_clean(val_loader, model):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = F.cross_entropy(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def eval_test_pgd(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target, _ in test_loader:
        data, target = data.cuda(), target.cuda()
        # or atk = torchattacks.PGD
        x_adv = data.detach() + torch.FloatTensor(*data.shape).uniform_(-args.epsilon, args.epsilon).cuda()

        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), target)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, data - args.epsilon), data + args.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        with torch.no_grad():
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    # print('Test PGD: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


if __name__ == '__main__':
    main()
