import torch
from torchvision.models import vgg16

from utils import prune, get_flops_and_params
# from strategy.pca_kmeans import get_remain_channel
from strategy.featureMapReconstruction import get_remain_channel
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("fun/train/vgg16_cifar10_CAWR_20221005_150722/vgg16.pth", map_location=device)
    # model = vgg16(num_classes=10)
    # model.to(device)
    # print(model)

    matrix_info_retention_ratio = 0.9

    in_remain_channel = [i for i in range(model.features[0].in_channels)]
    out_remain_channel = None

    intermediate_feature_map = torch.randn(4, 3, 32, 32).to(device)
    # feature
    for layer in tqdm(model.features):
        if isinstance(layer, torch.nn.Conv2d):
            out_remain_channel, intermediate_feature_map = get_remain_channel(layer, matrix_info_retention_ratio, intermediate_feature_map)
            prune(layer, out_remain_channel, in_remain_channel)
            in_remain_channel = out_remain_channel
    
    # classifier
    n = model.avgpool.output_size[0] * model.avgpool.output_size[0]
    in_remain_channel_ = []
    for x in in_remain_channel:
        in_remain_channel_ += [i for i in range(x * n, x * n + n)]
    prune(model.classifier[0], out_remain_channel=None, in_remain_channel=in_remain_channel_)

    # torch.save(model, "fun/train/vgg16_cifar10_CAWR_20221005_150722/pruned_vgg16_90.pth")

    print(model)
    print("==========================test==========================")
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    model.eval()
    print(model(dummy_input))
    # print(get_flops_and_params(model))


if __name__ == "__main__":
    main()



