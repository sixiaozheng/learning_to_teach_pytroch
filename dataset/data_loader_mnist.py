import torch
import torchvision
import torchvision.transforms as transforms
import os
import dataset.mnist


def data_loader_func(batch_sizes, path_to_dataset):
    """
    Preparing the transform, dataset, dataloader, etc.
    :param batch_sizes:
    :param epochs:
    :return: none
    """

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)

    # trainset = torchvision.datasets.CIFAR10(root=path_to_dataset, train=True,
    #                                         download=True, transform=transform_train)

    teacher_train_set = dataset.mnist.MNIST(root=path_to_dataset, transform=transform_train, split='teacher_train')
    teacher_train_loader = torch.utils.data.DataLoader(teacher_train_set, batch_size=batch_sizes,
                                                       shuffle=True, num_workers=0, drop_last = True)

    dev_set = dataset.mnist.MNIST(root=path_to_dataset, transform=transform_test, split='dev')
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_sizes,
                                             shuffle=False, num_workers=0, drop_last = True)

    student_train_set = dataset.mnist.MNIST(root=path_to_dataset, transform=transform_train, split='student_train')
    student_train_loader = torch.utils.data.DataLoader(student_train_set, batch_size=batch_sizes,
                                                       shuffle=True, num_workers=0, drop_last = True)

    test_set = dataset.mnist.MNIST(root=path_to_dataset, transform=transform_test, split='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_sizes,
                                              shuffle=False, num_workers=0, drop_last = True)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')

    dataloader = {'teacher_train_loader': teacher_train_loader,
                  'dev_loader': dev_loader,
                  'student_train_loader': student_train_loader,
                  'test_loader': test_loader,
                  'classes': classes}

    return dataloader


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#
#     # functions to show an image
#     def imshow_2(img):
#         img = img / 2 + 0.5  # unnormalize
#         npimg = img.numpy()
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.show()
#
#
#     def imshow(inp, title=None):
#         """Imshow for Tensor."""
#         inp = inp.numpy().transpose((1, 2, 0))
#         mean = np.array((0.4914, 0.4822, 0.4465))
#         std = np.array((0.2023, 0.1994, 0.2010))
#         inp = std * inp + mean
#         inp = np.clip(inp, 0, 1)
#         plt.imshow(inp)
#         plt.show()
#         if title is not None:
#             plt.title(title)
#         plt.pause(0.001)  # pause a bit so that plots are updated
#
#
#     # get some random training images
#     dataloader = data_loader_func(batch_sizes=50, path_to_dataset='../data')
#     dataiter = iter(dataloader['student_train_loader'])
#     images, labels = dataiter.next()
#     print(' '.join('%5s' % dataloader['classes'][labels[j]] for j in range(50)))
#     # show images
#     imshow_2(torchvision.utils.make_grid(images))
#     # print labels
#
#     # class_img_list = [0]*10
#     # total = 0
#     # for images, labels in trainloader:
#     #     for label in labels:
#     #         class_img_list[label.item()]+=1
#     #     total+=labels.size(0)
#     #
#     # print(class_img_list)
#     # print(total)
#     #
#     # class_img_list = [0] * 10
#     # total = 0
#     # for images, labels in testloader:
#     #     for label in labels:
#     #         class_img_list[label.item()] += 1
#     #     total += labels.size(0)
#     #
#     # print(class_img_list)
#     # print(total)
