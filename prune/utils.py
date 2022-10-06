import torch

from thop import profile


def prune(layer, out_remain_channel, in_remain_channel=None):
    if isinstance(layer, torch.nn.Conv2d):
        if in_remain_channel is None:
            raise ValueError("missing input mask")
        # modify layer's metadata
        layer.in_channels = len(in_remain_channel)
        layer.out_channels = len(out_remain_channel)
        # inherit weight
        w = layer.weight.data.clone()[out_remain_channel]
        w = w[:, in_remain_channel]
        layer.weight.data = w
        # inherit bias
        if layer.bias is not None:
            b = layer.bias.data.clone()[out_remain_channel]
            layer.bias.data = b
    elif isinstance(layer, torch.nn.BatchNorm2d):
        # modify layer's metadata
        layer.num_features = len(out_remain_channel)
        # inherit weight and bias
        layer.weight.data = layer.weight.data.clone()[out_remain_channel]
        layer.bias.data = layer.bias.data.clone()[out_remain_channel]
        # inherit mean and varience
        layer.running_mean.data = layer.running_mean.data.clone()[out_remain_channel]
        layer.running_var.data = layer.running_var.data.clone()[out_remain_channel]
    elif isinstance(layer, torch.nn.Linear):
        if in_remain_channel is None:
            raise ValueError("missing input mask")
        # modify layer's metadata
        layer.in_features = len(in_remain_channel)
        # inherit weight
        layer.weight.data = layer.weight.data.clone()[:, in_remain_channel]
    else:
        raise TypeError("layer should be Conv, BatchNorm or Linear")


def get_flops_and_params(model, dummy_input_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, dummy_input_size, dummy_input_size).to(device)
    flops, params = map(int, profile(model, inputs=(dummy_input,)))
    return flops, params



