# inspired by https://cloud.tencent.com/developer/article/1893403
import torch
from copy import deepcopy
from random import randint


def shuffle(ts):
    size = ts.size()
    ts = ts.reshape(-1)
    for i in range(ts.size(0)):
        j = randint(i, ts.size(0) - 1)
        ts[i], ts[j] = ts[j].item(), ts[i].item()
    ts = ts.reshape(size)
    return ts


def getAvgVar(ts, std_ts):
    squared_diff = (ts - std_ts) ** 2   # 当前tensor和标准tensor每个元素的平方偏差
    sum_squared_diff = [torch.sum(squared_diff[i]).item() for i in range(squared_diff.size(0))] # 每个特征图的偏差平方和
    avg_var = sum(sum_squared_diff) / len(sum_squared_diff) # 一个batch的平均偏差平方和
    return avg_var


def get_channel_importance(layer, dummy_input=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dummy_input is None:
        dummy_input = torch.randn(4, layer.in_channels, 32, 32).to(device)
    
    layer.eval()
    std_y = layer(dummy_input)  # 当前卷积层的标准输出特征图

    # 逐个通道shuffle，计算其batch内的平均偏差平方和
    # 平均偏差平方和象征对应通道的重要性
    channel_importance = []
    for i in range(layer.out_channels):
        tmp = deepcopy(layer)
        # tmp.weight.data[i] *= 0   # 权重置零，相当于通道无效化
        # if tmp.bias is not None:
        #     tmp.bias.data[i] = 0
        tmp.weight.data[i] = shuffle(tmp.weight.data[i])    # 权重shuffle，相当于打乱通道
        tmp.eval()
        y = tmp(dummy_input)
        var = getAvgVar(y, std_y)
        channel_importance.append((i, var))
    return channel_importance, std_y


def get_remain_channel(layer, importance_retention_ratio=0.9, dummy_input=None):
    '''
    layer: 当前卷积层
    importance_retention_ration: 重要性保留比例
    '''
    # 获取当前卷积层的所有通道重要性
    channel_importance, std_y = get_channel_importance(layer, dummy_input)
    
    # 将通道重要性转换为通道重要性比例，并且降序排序
    total_importance = 0
    for x in channel_importance:
        total_importance += x[1]
    channel_importance_ratio = [(x[0], x[1] / total_importance) for x in channel_importance]
    channel_importance_ratio.sort(key=lambda x: -x[1])

    remain_channel = []
    if importance_retention_ratio < 1:
        # 保留若干个重要性比例最大的通道，使得它们的重要性比例之和不小于给定阈值
        cur_ratio = 0
        for x in channel_importance_ratio:
            remain_channel.append(x[0])
            cur_ratio += x[1]
            if cur_ratio >= importance_retention_ratio:
                break
    else:
        for i in range(importance_retention_ratio):
            remain_channel.append(channel_importance_ratio[i][0])
    remain_channel.sort()
    return remain_channel, std_y



