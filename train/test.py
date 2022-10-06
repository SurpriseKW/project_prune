import torch
import get_dataloader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, testLoader= get_dataloader.get_cifar10(batch_size=32)

    model = torch.load("fun/train/resnet18_CAWR_20220926_133716/resnet18.pth", map_location=device)
    model.eval()

    correct = 0
    total = 0
    for images, labels in testLoader:
        images = images.to(device)
        labels = labels.to(device)

        y = model(images)
        
        _, pre = torch.max(y.data, 1)
        correct += (pre == labels).sum()
        total += len(labels)
    print("Acc: {}%({}/{})".format("%.2f" % (correct / total * 100), correct, total))


if __name__ == "__main__":
    main()