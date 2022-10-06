import os
import sys
import time
import logging

import torch
from torchvision.models import resnet50
from torchvision import transforms

import get_dataloader


def train():
    total_epochs = 300
    batch_size = 256
    num_workers = 8
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainLoader, testLoader = get_dataloader.get_cifar10(batch_size, num_workers)

    model_folder = sys.path[0] + os.sep + 'resnet50_cifar10_CAWR_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.mkdir(model_folder)
    logging.basicConfig(filename=model_folder + os.sep + 'log.log', level=logging.INFO)

    model = resnet50(num_classes=10)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0, T_0=total_epochs//3+1, T_mult=2)

    for epoch in range(total_epochs):
        # 训练
        model.train()
        for i, (images, labels) in enumerate(trainLoader):
            images = images.to(device)
            labels = labels.to(device)

            y = model(images)
            loss = loss_function(y, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                _, pre = torch.max(y.data, 1)
                correct = (pre == labels).sum()
                total = len(labels)
                msg = 'Epoch [{}/{}], Iteration [{}/{}], Loss {}, Acc {}/{}'.format(
                    epoch + 1, total_epochs,
                    i + 1, len(trainLoader),
                    '%.4f' % loss,
                    correct, total
                )
                logging.info(msg)
                # print(msg)
        
        # 每1个epoch测试一次
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

        msg = 'Epoch: [{}/{}], Acc: {}%({}/{})'.format(
            epoch + 1, total_epochs,
            '%.2f' % (correct / total * 100),
            correct, total
        )
        logging.info(msg)
        print(msg)

        scheduler.step()

        # 每50个epoch保存断点
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }
            ckpt_folder = model_folder + os.sep + 'checkpoint'
            if not os.path.exists(ckpt_folder):
                os.mkdir(ckpt_folder)
            torch.save(checkpoint, ckpt_folder + os.sep + 'epoch' + str(epoch + 1) + '.ckpt')
    torch.save(model, model_folder + os.sep + 'resnet50.pth')


if __name__ == "__main__":
    train()