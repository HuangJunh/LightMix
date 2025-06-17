"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import load_dataset.data_loader_cifar100 as data_loader
import os
import numpy as np
import math
import copy
from datetime import datetime
import multiprocessing
from utils import Utils
from template.drop import drop_path
from collections import OrderedDict
from template.tools import cal_flops_params

kernel_sizes = [0, 3, 5, 7]

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.
class SELayer(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SELayer, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = Hswish()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class ECALayer(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(in_channels, 2)+b)/gamma))
        k = t if t%2 else t+1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, drop_connect_rate=0.0, affine=True):
        super(ConvBNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=int((kernel_size - 1) / 2) * 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True)
        )
        self.drop_connect_rate = drop_connect_rate
        self.stride = stride

    def forward(self, x):
        out = self.op(x)
        if self.drop_connect_rate > 0:
            out = drop_path(out, drop_prob=self.drop_connect_rate, training=self.training)
        return out

class DW_Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, act_func=None):
        super(DW_Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding=int((kernel_size - 1) / 2), bias=False, groups=C_in),
        )
        if act_func:
            self.op1 = nn.Sequential(
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=True) if act_func == 'relu' else Hswish(inplace=True))
        else:
            self.op1 = nn.Sequential()

    def forward(self, x):
        out = self.op(x)
        out = self.op1(out)
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def __repr__(self):
        return '%s(groups=%d)' % (self.__class__.__name__, self.groups)

    def forward(self, x):
        'Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )

class MixedScaleModule(nn.Module):
    def __init__(self, C_in, C_out, conn_encoding, stride, act_func, is_test=0):
        super(MixedScaleModule, self).__init__()
        self.is_test = is_test

        valid_conn = []
        valid_kernel_sizes = []
        for i, conn in enumerate(conn_encoding):
            if conn > 0:
                valid_conn.append(conn)  # [2,1]
                valid_kernel_sizes.append(kernel_sizes[i])  # [3,7]

        num_conn = len(valid_conn)
        splits = self.split_integer(num_conn, 2)

        branch1_kernel_size = valid_kernel_sizes[:splits[0]]
        branch1_partition = valid_conn[:splits[0]]
        branch2_kernel_size = valid_kernel_sizes[splits[0]:]
        branch2_partition = valid_conn[splits[0]:]

        gcd1 = self.gcd(*branch1_partition)
        gcd2 = self.gcd(*branch2_partition)
        simplified_branch1_partition = list(np.asarray(branch1_partition) // gcd1)
        simplified_branch2_partition = list(np.asarray(branch2_partition) // gcd2)
        in_splits1_one = self.split_integer(C_in, sum(simplified_branch1_partition))
        in_splits2_one = self.split_integer(C_in, sum(simplified_branch2_partition))
        in_splits1, in_splits2 = [], []

        k = 0
        for factor in simplified_branch1_partition:
            in_splits1.append(sum(in_splits1_one[k:k + factor]))
            k = k + factor
        k = 0
        for factor in simplified_branch2_partition:
            in_splits2.append(sum(in_splits2_one[k:k + factor]))
            k = k + factor

        branch1_1, branch1_2 = [], []
        for idx, (k, in_ch, out_ch) in enumerate(zip(branch1_kernel_size, in_splits1, in_splits1)):
            if k == 0:
                branch1_1.append(nn.Sequential(Identity(), nn.MaxPool2d(3, stride=2, padding=1) if stride > 1 else nn.Sequential()))
            else:
                branch1_1.append(DW_Conv(in_ch, out_ch, k, stride, act_func=act_func))

        for idx, (k, in_ch, out_ch) in enumerate(zip(branch2_kernel_size, in_splits2, in_splits2)):
            branch1_2.append(DW_Conv(in_ch, out_ch, k, stride, act_func=act_func))

        self.branch1_1 = nn.ModuleList(branch1_1)
        self.branch1_2 = nn.ModuleList(branch1_2)

        self.in_split1 = in_splits1
        self.in_split2 = in_splits2

    def split_channels(self, num_chan, num_groups):
        split = [num_chan // num_groups for _ in range(num_groups)]
        split[0] += num_chan - sum(split)
        return split

    def split_integer(self, m, n):
        assert n > 0
        quotient = int(m / n)
        remainder = m % n
        if remainder > 0:
            return [quotient] * (n - remainder) + [quotient + 1] * remainder
        if remainder < 0:
            return [quotient - 1] * -remainder + [quotient] * (n + remainder)
        return [quotient] * n

    def gcd(self, *num):
        # calculate the greatest common divisor gcd
        gcd1 = []
        for i in range(1, sorted(num)[0] + 1):
            for index, j in enumerate(num):
                if j % i == 0:
                    if (index + 1) == len(num):
                        gcd1.append(i)
                        break
                    continue
                else:
                    break
        if not gcd1:
            return 1
        else:
            return sorted(gcd1)[-1]

    def forward(self, x):
        x_split1_1 = torch.split(x, self.in_split1, 1)
        x_split1_2 = torch.split(x, self.in_split2, 1)

        out1 = torch.cat([c(x) for x, c in zip(x_split1_1, self.branch1_1)], 1)
        out2 = torch.cat([c(x) for x, c in zip(x_split1_2, self.branch1_2)], 1)
        # out = torch.cat([out1, out2], 1)
        out = out1 + out2
        return out

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_in_channels = self.split_channels(in_channels, self.num_groups)
        self.split_out_channels = self.split_channels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def split_channels(self, num_chan, num_groups):
        split = [num_chan // num_groups for _ in range(num_groups)]
        split[0] += num_chan - sum(split)
        return split

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = self.split_channels(channels, self.num_groups)

        branch1_2 = []
        for i in range(self.num_groups):
            if kernel_size[i] == 0:
                branch1_2.append(nn.Sequential(Identity(), nn.MaxPool2d(3, stride=2, padding=1) if stride > 1 else nn.Sequential()))
            else:
                branch1_2.append(DW_Conv(self.split_channels[i], self.split_channels[i], kernel_size[i], stride=stride, act_func='relu'))
        self.branch1_2 = nn.ModuleList(branch1_2)

    def split_channels(self, num_chan, num_groups):
        split = [num_chan // num_groups for _ in range(num_groups)]
        split[0] += num_chan - sum(split)
        return split

    def forward(self, x):
        if self.num_groups == 1:
            return self.branch1_2[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.branch1_2, x_split)]
        x = torch.cat(x, dim=1)
        return x

class Block(nn.Module):
    def __init__(self, C_in1, C_in2, C_mid, C_out, expand_ksize, project_ksize, stride, act_func, conn_encoding, attention, drop_connect_rate, is_test):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_connect_rate = float(drop_connect_rate)
        attention = float(attention)
        conn_encoding = list(map(int, conn_encoding.split()))
        expand_ksize = list(map(int, expand_ksize.split()))
        project_ksize = list(map(int, project_ksize.split()))

        self.C_in1 = C_in1
        self.C_in2 = C_in2

        if C_in1 == C_mid or sum([1 for conn in conn_encoding if conn > 0]) == 1:
            C_exp = C_mid
            valid_kernel_sizes = []
            for i, conn in enumerate(conn_encoding):
                if conn > 0:
                    valid_kernel_sizes.append(kernel_sizes[i])
            if C_in1 == C_mid:
                self.op1 = nn.Sequential()
            else:
                self.op1 = nn.Sequential(
                    GroupedConv2d(C_in1, C_exp, expand_ksize),
                    nn.BatchNorm2d(C_exp, affine=True),
                    nn.ReLU(inplace=True) if act_func == 'relu' else Hswish(inplace=True),
                )
            self.module = MDConv(C_exp, valid_kernel_sizes, self.stride)
        else:
            C_exp = C_mid
            self.op1 = nn.Sequential(
                GroupedConv2d(C_in1, C_exp, expand_ksize),
                nn.BatchNorm2d(C_exp, affine=True),
                nn.ReLU(inplace=True) if act_func == 'relu' else Hswish(inplace=True),
            )
            self.module = MixedScaleModule(C_exp, C_mid, conn_encoding, self.stride, act_func)

        if attention:
            self.attention = SELayer(C_mid, C_in1, attention)
            # self.attention = ECALayer(C_mid)
        else:
            self.attention = nn.Sequential()

        self.op2 = nn.Sequential(
            # nn.Conv2d(C_mid, C_out, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            GroupedConv2d(C_mid, C_out, project_ksize),
            nn.BatchNorm2d(C_out, affine=True),
        )

        dw_kernel_size = 3
        if (C_in1 == C_out and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(C_in1, C_in1, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2,
                          groups=C_in1, bias=False),
                nn.BatchNorm2d(C_in1),
                nn.Conv2d(C_in1, C_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out),
            )

        if not self.C_in2 == 0:
            self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x, drop_connect_rate):
        if not self.C_in2 == 0:
            x1, x2 = torch.split(x, [self.C_in1, self.C_in2], dim=1)
        else:
            x1 = x
        # 1st 1*1 conv layer
        out = self.op1(x1)
        # Mixed scale module
        out = self.module(out)
        # Squeeze-and-excitation
        out = self.attention(out)
        # 2nd 1*1 conv layer
        out = self.op2(out)

        if drop_connect_rate > 0:
            out = drop_path(out, drop_prob=drop_connect_rate, training=self.training)
        # if self.residual_connection:
        out = out + self.shortcut(x1)

        if not self.C_in2 == 0:
            out = torch.cat([out, x2], dim=1)
            out = self.shuffle(out)

        return out

class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.actf = nn.ReLU(inplace=True)
        #generated_init

        self._initialize_weights()

    def forward(self, x):
        #generate_forward


        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.actf(self.conv_end(out))
        out = out.view(out.size(0), -1)

        out = F.dropout(out, p=0.2, training=self.training)
        out = self.linear(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')


class TrainModel(object):
    def __init__(self, is_test, args):
        self.args = args
        if is_test:
            full_trainloader = data_loader.get_train_loader('../datasets/CIFAR100_data', batch_size=self.args.batch_size, augment=True,shuffle=True, random_seed=1000, show_sample=False,num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('../datasets/CIFAR100_data', batch_size=self.args.batch_size, shuffle=False,num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        else:
            trainloader, validate_loader = data_loader.get_train_valid_loader('../datasets/CIFAR100_data', batch_size=self.args.batch_size_search, augment=True, subset_size=1, valid_size=0.1, shuffle=True,random_seed=1000, show_sample=False,num_workers=4, pin_memory=True)
            self.trainloader = trainloader
            self.validate_loader = validate_loader


        net = EvoCNNModel()
        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()

        inputs = torch.randn(1, 3, 32, 32)
        self.flops, self.params = cal_flops_params(net, input_size=inputs.shape)

        net = net.cuda()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion.cuda()
        self.best_acc = best_acc
        self.best_epoch = 0
        self.file_id = os.path.basename(__file__).split('.')[0]

        if not is_test:
            self.log_record('Start training', first_time=True)
        else:
            self.log_record('Start testing', first_time=True)

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f'% (epoch+1, running_loss/total, (correct/total)))


    def final_train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.full_trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))


    def validate(self, epoch):
        self.net.eval()

        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = correct / total
        self.log_record('Validate-Epoch:%4d,  Validate-Loss:%.4f, Acc:%.4f'%(epoch + 1, test_loss/total, correct/total))


    def hooklogdet(self, K, N_A):
        s, ld = np.linalg.slogdet(K)
        # s, ld = np.linalg.slogdet(K / N_A)  # normalization
        return ld


    def counting_forward_hook(self, module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        self.net.K = self.net.K + K.cpu().numpy() + K2.cpu().numpy()
        # print('self.net.K:', self.net.K)
        self.net.N_A = self.net.K[0][0]

    def counting_backward_hook(self, module, inp, out):
        module.visited_backwards = True

    def process(self):
        self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))

        self.net.K = np.zeros((self.args.batch_size_search, self.args.batch_size_search))
        self.net.N_A = 0

        for name, module in self.net.named_modules():
            # print('name:',name)
            if 'ReLU' in str(type(module)):
                module.register_forward_hook(self.counting_forward_hook)
                module.register_backward_hook(self.counting_backward_hook)

        # self.net = self.net.cuda()
        data_iterator = iter(self.trainloader)
        x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.cuda()
        x, target = x.cuda(), target.cuda()
        self.net(x2.cuda())
        # print('final self.net.K:', self.net.K)
        score = self.hooklogdet(self.net.K, self.net.N_A)
        self.best_acc = score
        torch.cuda.empty_cache()
        return self.best_acc, self.params, self.flops

    def adjust_lr_warmup(self, ini_lr, lr_rate, optimizer, epoch):
        lr = ini_lr + ((lr_rate - ini_lr) / 5) * epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def process_test(self):
        drop_connect_rate = self.args.drop_connect_rate
        total_epoch = self.args.epoch_test
        lr_rate = 0.05
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch-5)

        lr_rate_init = 0.0001
        optimizer_ini = optim.SGD(self.net.parameters(), lr=lr_rate_init, momentum=0.9, weight_decay=self.args.weight_decay)

        self.test(0)
        for p in range(total_epoch):
            self.net.drop_connect_rate = drop_connect_rate * p / total_epoch
            if p < 5:
                self.final_train(p, optimizer_ini)
                self.test(p)
                # scheduler_ini.step()
                self.adjust_lr_warmup(lr_rate_init, lr_rate, optimizer_ini, p+1)
            else:
                self.final_train(p, optimizer)
                self.test(p)
                scheduler.step()
        return self.best_acc, self.params, self.flops

    def test(self,p):
        self.net.eval()

        if p == 0:
            self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))

        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            torch.save(self.net.state_dict(), './trained_models/best_CNN7.pt')
            self.best_acc = correct / total
        self.log_record('Test-Loss:%.4f, Acc:%.4f' % (test_loss / total, correct / total))

class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, args):
        torch.cuda.empty_cache()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        best_acc = 0.0
        params = 1e9
        flops = 10e9
        try:
            m = TrainModel(is_test, args)
            m.log_record('Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
            if is_test:
                best_acc, params, flops = m.process_test()
            else:
                best_acc, params, flops = m.process()

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.4f'%best_acc)
            m.log_record('Finished-Err:%.4f' % (1-best_acc))
            m.log_record('#Parameters:%d' % params)
            m.log_record('#FLOPs:%d' % flops)
"""
