#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import math


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=20, help="number of users: K")
    parser.add_argument('--select_frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--local_epochs', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_batch_size', type=int, default=32, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--dataset', type=str, default='mnist', help="train-test datasplit.py type, user or sample")
    parser.add_argument('--strategy', type=str, default='fed_avg', help="aggregate algorithm")
    parser.add_argument('--eval_batch_size', type=int, default=128, help="test batch size")

    # attacker  arguments
    parser.add_argument('--malicious_frac', type=float, default=0.3, help='percentage of malicious clients')
    parser.add_argument('--label_flipping_enable', type=bool, default=False,
                        help='whether to enable the label_flipping attack')
    parser.add_argument('--backdoor_enable', type=float, default=False, help='whether to enable the backdoor attack')
    parser.add_argument('--source_label', type=int, default=0)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--inert_enable', type=bool, default=True, help='whether to enable the inert attack')
    parser.add_argument('--gauss_mean', type=float, default=0.0)
    parser.add_argument('--gauss_std', type=float, default=math.sqrt(30))
    parser.add_argument('--gradient_scale_enable', type=bool, default=False,
                        help='whether to enable the gradient scale attack')
    parser.add_argument('--scale', type=float, default=-1.0)

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--split', type=str, default='mnist', help="name of split")
    # parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--partition', type=str, default='iid', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
