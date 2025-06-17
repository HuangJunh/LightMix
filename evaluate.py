# !/usr/bin/python
# -*- coding: utf-8 -*-

from utils import Log
from multiprocessing import Process
import importlib
import sys
import numpy as np
import copy, torch
from template.tools import cal_flops_params

def fitnessEvaluate(net, trainloader, args):

    if args.dataset.__contains__('cifar'):
        inputs = torch.randn(1, 3, 32, 32)
    else:
        inputs = torch.randn(1, 3, 224, 224)
    n_flops, n_parameters = cal_flops_params(net, input_size=inputs.shape)

    net = net.cuda()
    net.K = np.zeros((args.batch_size_search, args.batch_size_search))

    def hooklogdet(K, labels=None):
        s, ld = np.linalg.slogdet(K)
        return ld

    def counting_forward_hook(module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()


    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

    data_iterator = iter(trainloader)
    x, target = next(data_iterator)
    x2 = torch.clone(x)
    x2 = x2.cuda()
    x, target = x.cuda(), target.cuda()
    net(x2.cuda())
    score = hooklogdet(net.K, target)
    err = 1 - score
    return err, n_parameters, n_flops


def fitnessTest(file_name, curr_gen, is_test, args):
    Log.info('Begin to train %s' % (file_name))
    module_name = 'scripts.%s' % (file_name)
    if module_name in sys.modules.keys():
        Log.info('Module:%s has been loaded, delete it' % (module_name))
        del sys.modules[module_name]
        _module = importlib.import_module('.', module_name)
    else:
        _module = importlib.import_module('.', module_name)
    _class = getattr(_module, 'RunModel')
    cls_obj = _class()
    p = Process(target=cls_obj.do_work, args=('%d' % (args.gpu), curr_gen, file_name, is_test, args))
    p.start()


