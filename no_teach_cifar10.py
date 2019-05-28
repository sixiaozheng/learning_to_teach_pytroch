import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torchvision

from network.teacher_model.teacher_mlp import teacher_mlp
from network.student_model.resnet import resnet32
from network.student_model import resnet
from network.student_model.MLP import MLP, weights_init
from dataset.data_loader import data_loader_func
from utils.setseed import setup_seed
from itertools import count
from scipy.stats import rankdata
import math
import argparse
import os
import copy
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm


def student_learning_rate_scheduler_cifar10(optimizer, iterations):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations == 32000:
        lr = lr * 0.1
        print ('Adjust learning rate to : ', lr)
    elif iterations == 48000:
        lr = lr * 0.1
        print ('Adjust learning rate to : ', lr)
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_student():
    num_effective_data = 0
    since = time.time()
    student_update = 0
    for epoch in tqdm(range(config['epochs']), desc='Epochs'):

        train_running_loss = 0.0
        train_running_corrects = 0
        train_total = 0

        for idx, (inputs, labels) in enumerate(dataloader['student_train_loader']):
            student_model.train()

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = student_model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_total += inputs.size(0)
            num_effective_data += inputs.size(0)
            train_running_loss += loss.item() * inputs.size(0)
            train_running_corrects += torch.sum(predicted == labels.data)
            student_update += 1
            student_learning_rate_scheduler_cifar10(optimizer, student_update)

            student_model.eval()
            test_running_loss = 0.0
            test_running_corrects = 0
            test_total = 0

            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(dataloader['test_loader']):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = student_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    test_total += inputs.size(0)
                    test_running_loss += loss.item() * inputs.size(0)
                    test_running_corrects += torch.sum(predicted == labels.data)

                test_epoch_loss = test_running_loss / test_total
                test_epoch_acc = test_running_corrects.double() / test_total

                writer.add_scalar('test_acc', test_epoch_acc, num_effective_data)
                writer.add_scalar('test_loss', test_epoch_loss, num_effective_data)

                print('test_acc: {:.4f}, test_loss: {:.4f}'.format(test_epoch_acc, test_epoch_loss))

        train_epoch_loss = train_running_loss / train_total
        train_epoch_acc = train_running_corrects.double() / train_total

        writer.add_scalar('train_acc', train_epoch_acc, num_effective_data)
        writer.add_scalar('train_loss', train_epoch_loss, num_effective_data)

        print('train_acc: {:.4f}, train_loss: {:.4f}'.format(train_epoch_acc, train_epoch_loss))

    time_elapsed = time.time() - since
    print('='*20)
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed//3600, time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning to teach')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    # set seed
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'tau': 0.80,
        'max_iter': 60000,
        'batch_size': 128,
        'max_non_increasing_steps': 10,
        'num_classes': 10,
        'max_loss': 2.47,
        'epochs': 180

    }

    student_model = resnet32().to(device)

    dataloader = data_loader_func(batch_sizes=config['batch_size'], path_to_dataset='./data')

    optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter('./runs/no_teach_cifar10')

    print('Training the student starts....................')
    train_student()

    model_save_dir = './result/student'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(student_model.state_dict(), os.path.join(model_save_dir, 'student_no_teach_cifar10.pth'))
    # train_teacher(teacher_model, student_model, dataloader, device, config)