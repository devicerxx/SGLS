from __future__ import absolute_import
from .loss import CrossEntropy, AT, SGLS


def get_loss(args, labels=None, num_classes=10):
    if args.loss == 'ce':
        criterion = CrossEntropy()
    elif args.loss == 'at':
        criterion = AT(step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps)
    elif args.loss == 'sgls':
        criterion = SGLS(labels, num_classes=num_classes, momentum=args.sgls_alpha , es=args.sgls_es,
                    step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps)
    else:
        raise KeyError("Loss `{}` is not supported.".format(args.loss))

    return criterion
