import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision

from network.teacher_model.teacher_mlp import teacher_mlp
from network.student_model.resnet import resnet32
from network.student_model import resnet
from dataset.data_loader import data_loader_func

from itertools import count

from scipy.stats import rankdata
import math


def state_func(state_config):
    inputs = state_config['inputs']
    labels = state_config['labels']
    n_samples = inputs.size(0)
    student_model = state_config['model']
    # model = model.to(self.device)
    criterion = nn.CrossEntropyLoss()
    outputs = student_model(inputs)
    train_loss = criterion(outputs, labels)

    val_loss, val_acc = val_student(student_model, state_config['dataloader']['dev_loader'], state_config['device']) #model, dataloader, device

    state_config['training_loss_history'].append(train_loss.item())
    average_train_loss = sum(state_config['training_loss_history']) / len(state_config['training_loss_history'])
    state_config['best_loss_on_dev'] = val_loss if val_loss < state_config['best_loss_on_dev'] else state_config[
        'best_loss_on_dev']

    log_P_y = torch.log(outputs[range(n_samples), labels.data]).reshape(-1, 1)

    mask = torch.ones(inputs.size(0), state_config['num_class']).to(state_config['device'])
    mask[range(n_samples), labels.data] = 0
    margin_value = (outputs[range(n_samples), labels.data] - torch.max(mask * outputs, 1)[0]).reshape(-1, 1)

    state_feactues = torch.zeros((inputs.size(0), 25)).to(device)
    state_feactues[range(n_samples), state_config['labels'].data] = 1
    state_feactues[range(n_samples), 10] = state_config['i_iter'] / config['max_iter']
    state_feactues[range(n_samples), 11] = average_train_loss / config['max_loss']
    state_feactues[range(n_samples), 12] = state_config['best_loss_on_dev'] / config['max_loss']
    state_feactues[range(n_samples), 13:23] = outputs
    state_feactues[range(n_samples), 23] = torch.from_numpy(
        rankdata(-log_P_y.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)
    state_feactues[range(n_samples), 24] = torch.from_numpy(
        rankdata(margin_value.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)

    del outputs
    return state_feactues


def train_student(config):
    model = config['model']
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    inputs = config['inputs'].to(config['device'])
    labels = config['labels'].to(config['device'])

    # zero the parameter gradients
    optimizer = config['optimizer']
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    del outputs
    # print statistics
    total += inputs.size(0)
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(predicted == labels.data)

    train_loss = running_loss / total
    train_acc = running_corrects.double() / total

    return train_loss, train_acc


def val_student(model, dataloader, device):
    # dataloader = config['dataloader']
    # model = self.student_model
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            del outputs
            # statistics
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)

        val_loss = running_loss / total
        val_acc = running_corrects.double() / total

    return val_loss, val_acc


def student_lr_scheduler(optimizer, iterations):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations == 32000:
        lr = lr * 0.1
        print('Adjust learning rate to : ', lr)
    elif iterations == 48000:
        lr = lr * 0.1
        print('Adjust learning rate to : ', lr)
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def teacher_lr_scheduler(optimizer, iterations):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations % 50 == 0:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_teacher(config):
    i_iter = config['i_iter']
    max_iter = config['max_iter']
    reward_T_histtory = config['reward_T_histtory']
    rewards_baseline = config['rewards_baseline']
    i_episode = config['i_episode']
    saved_log_probs = config['saved_log_probs']
    optimizer = config['optimizer']
    teacher_updates = config['teacher_updates']

    optimizer.zero_grad()

    reward_T = -math.log(i_iter / max_iter)
    reward_T_histtory.append(reward_T)
    rewards_baseline = rewards_baseline + (reward_T - rewards_baseline) / (i_episode)
    policy_loss = -torch.cat(saved_log_probs).sum() * (reward_T - rewards_baseline)

    policy_loss.backward()
    optimizer.step()

    teacher_updates += 1
    teacher_lr_scheduler(optimizer, teacher_updates)

    last_reward = 0.0 if len(reward_T_histtory) == 0 else reward_T_histtory[-1]
    if abs(last_reward - reward_T) < 0.01:
        config['non_increasing_steps'] += 1
    else:
        config['non_increasing_steps'] = 0

    print('acc >= 0.8 at {}， reward_T: {}'.format(i_iter, reward_T))


def train_teacher(teacher_model, student_model, dataloader, device, config):
    training_loss_history = []
    val_loss_history = []
    num_steps_to_achieve = []
    teacher_updates = 0
    non_increasing_steps = 0
    student_updates = 0
    best_loss_on_dev = config['max_loss']
    # saved_log_probs = []

    for i_episode in count(1):
        input_pool = []
        label_pool = []
        count_sampled = 0
        i_iter = 0

        while i_iter < config['max_iter']:  # Don't infinite loop while learning
            # for i_iter in range(self.config['max_iter']):
            # collect training batch
            for idx, (inputs, labels) in enumerate(dataloader['teacher_train_loader']):

                state_config = {
                    'inputs': inputs.to(device),
                    'labels': labels.to(device),
                    'num_class': config['num_classes'],
                    'i_iter': i_iter,
                    'training_loss_history': training_loss_history,
                    'best_loss_on_dev': best_loss_on_dev,
                    'model':student_model,
                    'dataloader': dataloader,
                    'device': device
                }
                state = state_func(state_config)

                action_prb = teacher_model(state.detach())

                sampled_actions = torch.bernoulli(action_prb.data.squeeze())
                indices = torch.nonzero(sampled_actions)

                log_probs = []
                for idx, action in enumerate(sampled_actions):
                    if action.data == 1:
                        log_probs.append(torch.log(action_prb[idx]))
                    else:
                        log_probs.append(torch.log(1 - action_prb[idx]))
                teacher_model.saved_log_probs.append(torch.cat(log_probs).sum())

                if len(indices) == 0:
                    continue

                count_sampled += len(indices)
                selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                selected_labels = labels[indices.squeeze()].view(-1, 1)
                input_pool.append(selected_inputs)
                label_pool.append(selected_labels)

                i_iter += 1
                if count_sampled >= config['batch_size']:
                    break

            # generate data for train student model
            inputs = torch.cat(input_pool, 0)[:config['batch_size']]
            labels = torch.cat(label_pool, 0)[:config['batch_size']].squeeze()

            input_pool_tmp, label_pool_tmp = [], []
            input_pool_tmp.append(torch.cat(input_pool, 0)[config['batch_size']:])
            label_pool_tmp.append(torch.cat(label_pool, 0)[config['batch_size']:])
            input_pool = input_pool_tmp
            label_pool = label_pool_tmp
            count_sampled = input_pool[0].shape[0]

            train_student_config = {
                'inputs': inputs,
                'labels': labels,
                'device': device,
                'model': student_model,
                'optimizer': optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

            }
            train_loss, train_acc = train_student(train_student_config)
            training_loss_history.append(train_loss)
            i_iter += 1
            student_lr_scheduler(train_student_config['optimizer'], i_iter)

            # val_student_config = {'dataloader': self.dataloader['dev_loader']}
            val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
            best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev

            print('val loss: {}, val acc: {}'.format(val_loss, val_acc))
            if val_acc > config['tau'] or i_iter == config['max_iter']:
                num_steps_to_achieve.append(i_iter)

                break

        update_teacher_config = {
            'i_iter': i_iter,
            'max_iter': config['max_iter'],
            'non_increasing_steps': non_increasing_steps,
            'reward_T_histtory': teacher_model.reward_T_histtory,
            'rewards_baseline': teacher_model.rewards_baseline,
            'i_episode': i_episode,
            'teacher_updates': teacher_updates,
            'saved_log_probs': teacher_model.saved_log_probs,
            'optimizer': optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0)

        }
        update_teacher(update_teacher_config)

        # reinit the student model
        # self.student_model.apply(resnet._weights_init)

        if non_increasing_steps >= config['max_non_increasing_steps']:
            return


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'tau': 0.8,
        'max_iter': 60000,
        'batch_size': 128,
        'max_non_increasing_steps': 10,
        'num_classes': 10,
        'max_loss': 2.47
    }
    teacher_model = teacher_mlp().to(device)
    student_model = resnet32().to(device)
    dataloader = data_loader_func(batch_sizes=config['batch_size'], path_to_dataset='./data')


    train_teacher(teacher_model, student_model, dataloader, device, config)






