# -*- coding:utf-8 -*-

import configparser

import torch

from lib.utils import scaled_Laplacian, cheb_polynomial, get_adjacency_matrix


def get_backbones(config_filename, adj_filename, device):
    config = configparser.ConfigParser()
    config.read(config_filename)

    K = int(config['Training']['K'])
    num_of_weeks = int(config['Training']['num_of_weeks'])
    num_of_days = int(config['Training']['num_of_days'])
    num_of_hours = int(config['Training']['num_of_hours'])
    num_of_vertices = int(config['Data']['num_of_vertices'])

    adj_mx = get_adjacency_matrix(adj_filename, num_of_vertices)#获得邻接矩阵adj_mx
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = torch.tensor(cheb_polynomial(L_tilde, K), dtype=torch.float32).to(device)#.to(device)生成新的张量
    #生成最高3阶的切比雪夫多项式

    backbones1 = [
        {
            "K": K,
            "num_of_chev_filters": 64,#切比雪夫过滤器的数量
            "num_of_time_filters": 64,#时间节点过滤器的数量
            "time_conv_strides": num_of_weeks,#时间卷积跨步
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones2 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_days,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones3 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_hours,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    all_backbones = [
        backbones1,
        backbones2,
        backbones3
    ]

    return all_backbones
