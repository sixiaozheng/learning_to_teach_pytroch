# Generally, when you have to deal with image, text, audio or video data,
# you can use standard python packages that load data into a numpy array.
# Then you can convert this array into a torch.*Tensor.

# For images, packages such as Pillow, OpenCV are useful
# For audio, packages such as scipy and librosa
# For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

import numpy as np
import random
import argparse
from tqdm import tqdm
import time
import os
import copy

from network.student_model.resnet import resnet32
from dataset.data_loader import data_loader_func


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, criterion, optimizer, scheduler, writer, trainloader, testloader, start_epoch, epochs):  # scheduler,
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 20)

        scheduler.step()
        model.train()

        running_loss = 0.0
        running_corrects = 0
        total = 0

        # use prefetch_generator and tqdm for iterating through data
        pbar_train = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        start_time = time.time()

        for i, data in pbar_train:
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = time.time() - start_time

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute computation time and *compute_efficiency*
            process_time = time.time() - prepare_time - start_time
            pbar_train.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), epoch, epochs))

            # print statistics
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)

            start_time = time.time()

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

        # writer.add_text('text', '{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))
        writer.add_scalars('data/epoch', {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc}, epoch)

        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        # maybe do a test pass every x epochs
        val_epochs = 1
        if epoch % val_epochs == val_epochs - 1:
            # bring models to evaluation mode
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            with torch.no_grad():
                pbar_test = tqdm(enumerate(testloader, 0), total=len(testloader))
                for i, data in pbar_test:
                    # get the inputs
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # forward + backward + optimize
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # statistics
                    total += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predicted == labels.data)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val', epoch_loss, epoch_acc))

            # if epoch_acc >= 0.98:
            #     print('The accuracy exceed o.8 at {}-th epoch steps'.format(epoch))
            #     break

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            writer.add_scalars('data/val', {'val_loss': epoch_loss, 'val_acc': epoch_acc}, epoch)
            writer.add_scalar('data/best', best_acc, epoch)

            # save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'lr_scheduler': scheduler.state_dict()
            }, checkpoint_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test(model, testloader, classes):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        pbar_test = tqdm(enumerate(testloader, 0), total=len(testloader))
        for i, data in pbar_test:
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # plot a feature map
            # import matplotlib.pyplot as plt
            # def imshow(img):
            #     img = img / 2 + 0.5  # unnormalize
            #     npimg = img.cpu().numpy()
            #     plt.imshow(npimg)#np.transpose(npimg, (1, 2, 0)
            #     plt.show()
            #
            # out_1 = torch.squeeze(out_1[0, 0, :, :])
            # out_2 = torch.squeeze(out_2[0, 0, :, :])
            # print(out_1.shape)
            # print(out_2.shape)
            # imshow(out_1)
            # imshow(out_2)
            # imshow(torchvision.utils.make_grid(out_1))
            # imshow(torchvision.utils.make_grid(out_1))

            loss = criterion(outputs, labels)

            # statistics
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)

            c = (predicted == labels).squeeze()
            for i in range(inputs.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * running_corrects / total))

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', epoch_loss, epoch_acc))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    #
    # with torch.no_grad():
    #     for data in testloader:
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #
    #         total += inputs.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    #         c = (predicted == labels).squeeze()
    #         for i in range(inputs.size(0)):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #         100 * correct / total))
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network for Image Classification.')
    parser.add_argument('--batch_sizes', type=int, default=128, help='Batch sizes.')
    parser.add_argument('--epochs', type=int, default=328, help='Epochs.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Enable CUDA training')
    parser.add_argument('--ngpu', type=int, default=1, help='the num of gpu')
    parser.add_argument('--resume', action='store_true', default=False, help='load checkpoint if needed/ wanted.')
    parser.add_argument('--path_to_checkpoint', type=str, default='./log/checkpoint.pth',
                        help='the path of the checkpoint.')
    parser.add_argument('--path_to_dataset', type=str, default='./data',
                        help='the path of the dataset.')
    parser.add_argument('--path_to_log', type=str, default='./log',
                        help='the path of the dataset.')

    # saves arguments to config.txt file
    args = parser.parse_args()

    checkpoint_dir = './checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config_save_path = os.path.join(checkpoint_dir, 'config.txt')
    with open(config_save_path, 'w') as f:
        f.write(args.__str__())
        # json.dump(args.__str__(), handle, indent=4, sort_keys=False)

    setup_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() and (not args.no_cuda) else 'cpu')

    dataloader = data_loader_func(args.batch_sizes, args.path_to_dataset)

    # Or use the defined network
    model = resnet32()

    # print(model)
    model = model.to(device)
    if args.ngpu > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[164, 246], gamma=0.1)

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.path_to_checkpoint)  # custom method for loading last checkpoint
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # exp_lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        print("last checkpoint restored")

    if not os.path.exists(args.path_to_log):
        os.makedirs(args.path_to_log)
    writer = SummaryWriter(args.path_to_log)
    # writer.add_graph(model, torch.rand(args.batch_sizes, 3, 32, 32).to(device))

    images = next(iter(dataloader['teacher_train_loader']))[0]
    # mean = torch.tensor((0.1307,)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    # std = torch.tensor((0.3081,)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = torch.tensor((0.2023, 0.1994, 0.2010)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    images = images * std + mean
    print(images.shape)
    writer.add_image('Images', torchvision.utils.make_grid(images))

    start_time = time.time()
    model = train(model, criterion, optimizer, exp_lr_scheduler, writer, dataloader['teacher_train_loader'], dataloader['dev_loader'], start_epoch,
                  args.epochs)
    end_time = time.time()
    print('Finished Training')
    print('Training time:{}'.format(end_time - start_time))

    model_save_dir = './result/model'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model.state_dict(), os.path.join(model_save_dir, 'beat_acc_model.pth'))

    # Test the model

    model = resnet32()

    # model = torchvision.models.resnet34()
    # for param in model.parameters():
    #     param.requires_grad = Falses
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, out_features=10)

    model = model.to(device)
    if args.ngpu > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))

    model.load_state_dict(torch.load(os.path.join(model_save_dir, 'beat_acc_model.pth')))

    start_time = time.time()
    test(model, dataloader['test_loader'], dataloader['classes'])
    end_time = time.time()
    print('Finished Testing')
    print('Testing time:{}'.format(end_time - start_time))
