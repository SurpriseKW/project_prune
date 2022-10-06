import torch

from utils import prune, get_flops_and_params
from strategy.pca_kmeans import get_remain_channel


def prune_residualBlock(rb, in_remain_channel, matrix_info_retention_ratio):
    if type(rb).__name__ == "BasicBlock":
        out_remain_channel = prune_basicBlock(rb, in_remain_channel, matrix_info_retention_ratio)
    elif type(rb).__name__ == "Bottleneck":
        out_remain_channel = prune_bottleneck(rb, in_remain_channel, matrix_info_retention_ratio)
    return out_remain_channel


def prune_basicBlock(block, in_remain_channel, matrix_info_retention_ratio):
    tmp_remain_channel = get_remain_channel(block.conv1, matrix_info_retention_ratio)
    out_remain_channel = get_remain_channel(block.conv2, matrix_info_retention_ratio) if block.downsample is not None\
        else get_remain_channel(block.conv2, len(in_remain_channel))
    # conv1 & bn1
    prune(block.conv1, tmp_remain_channel, in_remain_channel)
    prune(block.bn1, tmp_remain_channel)
    # conv2 & bn2
    prune(block.conv2, out_remain_channel, tmp_remain_channel)
    prune(block.bn2, out_remain_channel)
    # downsample
    if block.downsample is not None:
        prune(block.downsample[0], out_remain_channel, in_remain_channel)
        prune(block.downsample[1], out_remain_channel)
    return out_remain_channel


def prune_bottleneck(block, in_remain_channel, matrix_info_retention_ratio):
    tmp_remain_channel_1 = get_remain_channel(block.conv1, matrix_info_retention_ratio)
    tmp_remain_channel_2 = get_remain_channel(block.conv2, matrix_info_retention_ratio)
    out_remain_channel = get_remain_channel(block.conv3, matrix_info_retention_ratio) if block.downsample is not None\
        else get_remain_channel(block.conv3, len(in_remain_channel))
    # conv1 & bn1
    prune(block.conv1, tmp_remain_channel_1, in_remain_channel)
    prune(block.bn1, tmp_remain_channel_1)
    # conv2 & bn2
    prune(block.conv2, tmp_remain_channel_2, tmp_remain_channel_1)
    prune(block.bn2, tmp_remain_channel_2)
    # conv3 & bn3
    prune(block.conv3, out_remain_channel, tmp_remain_channel_2)
    prune(block.bn3, out_remain_channel)
    # downsample
    if block.downsample is not None:
        prune(block.downsample[0], out_remain_channel, in_remain_channel)
        prune(block.downsample[1], out_remain_channel)
    return out_remain_channel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = torch.load("fun/train/resnet50_CAWR_20220927_125844/resnet50.pth", map_location=device)
    model = torch.load("fun/train/resnet18_CAWR_20220926_133716/resnet18.pth", map_location=device)

    matrix_info_retention_ratio = 0.8

    in_remain_channel = [i for i in range(model.conv1.out_channels)]
    out_remain_channel = None

    # conv1 and bn1 are preserved
    # layer1
    for block in model.layer1:
        in_remain_channel = prune_residualBlock(block, in_remain_channel, matrix_info_retention_ratio)
    # layer2
    for block in model.layer2:
        in_remain_channel = prune_residualBlock(block, in_remain_channel, matrix_info_retention_ratio)
    # layer3
    for block in model.layer3:
        in_remain_channel = prune_residualBlock(block, in_remain_channel, matrix_info_retention_ratio)
    # layer4
    for block in model.layer4:
        in_remain_channel = prune_residualBlock(block, in_remain_channel, matrix_info_retention_ratio)
    # fc
    prune(model.fc, out_remain_channel=None, in_remain_channel=in_remain_channel)

    print(model)
    print(get_flops_and_params(model, 32))
    print("==========================test==========================")
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    model.eval()
    print(model(dummy_input))


if __name__ == "__main__":
    main()



