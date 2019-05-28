import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torchvision

from network.teacher_model.teacher_mlp_a2c import TeacherNetwork
from network.student_model.resnet import resnet32, _weights_init
# from network.student_model.MLP import MLP, weights_init
from dataset.data_loader import data_loader_func
from utils.setseed import setup_seed
from utils.storage import Storage
from itertools import count
from scipy.stats import rankdata
import numpy as np
import math
import argparse
import os
from tensorboardX import SummaryWriter
from tqdm import trange
import time

def state_func(state_config):
    inputs = state_config['inputs']
    labels = state_config['labels']
    n_samples = inputs.size(0)
    student_model = state_config['model']
    max_loss = state_config['max_loss']
    best_loss_on_dev = state_config['best_loss_on_dev']

    outputs = student_model(inputs)
    average_train_loss = max_loss if len(state_config['training_loss_history']) == 0 else sum(
        state_config['training_loss_history']) / len(state_config['training_loss_history'])
    log_P_y = torch.log(outputs[range(n_samples), labels.data]).reshape(-1, 1)

    mask = torch.ones(inputs.size(0), state_config['num_class']).to(state_config['device'])
    mask[range(n_samples), labels.data] = 0
    margin_value = (outputs[range(n_samples), labels.data] - torch.max(mask * outputs, 1)[0]).reshape(-1, 1)

    state_feactues = torch.zeros((inputs.size(0), 25)).to(device)
    state_feactues[range(n_samples), state_config['labels'].data] = 1
    state_feactues[range(n_samples), 10] = state_config['student_iter'] / state_config['max_iter']
    state_feactues[range(n_samples), 11] = average_train_loss / max_loss
    state_feactues[range(n_samples), 12] = best_loss_on_dev / max_loss
    state_feactues[range(n_samples), 13:23] = outputs
    state_feactues[range(n_samples), 23] = torch.from_numpy(
        rankdata(-log_P_y.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)
    state_feactues[range(n_samples), 24] = torch.from_numpy(
        rankdata(margin_value.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)

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
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            # statistics
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)

        val_loss = running_loss / total
        val_acc = running_corrects.double() / total

    return val_loss, val_acc


def student_lr_scheduler_cifar10(optimizer, iterations):
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


# def student_lr_scheduler_mnist(optimizer, iterations):
#     lr = 0.0
#     for param_group in optimizer.param_groups:
#         lr = param_group['lr']
#         break
#     if iterations % 50 == 0:
#         lr = lr * 0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


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
    rewards = config['rewards']
    rewards_baseline = config['rewards_baseline']
    i_episode = config['i_episode']
    saved_log_probs = config['saved_log_probs']
    optimizer = config['optimizer']
    reward_T_histtory = config['reward_T_histtory']

    optimizer.zero_grad()

    reward_T = rewards[-1]
    # reward_T = -math.log(i_iter / max_iter)
    # reward_T_histtory.append(reward_T)
    # reward_T =
    rewards_baseline = rewards_baseline + (reward_T - rewards_baseline) / (i_episode + 1)
    policy_loss = -torch.cat(saved_log_probs).sum() * (reward_T - rewards_baseline)

    policy_loss.backward()
    optimizer.step()

    teacher_lr_scheduler(optimizer, i_episode)

    last_reward = 0.0 if len(reward_T_histtory) == 0 else reward_T_histtory[-1]
    if abs(last_reward - reward_T) < 0.01:
        config['non_increasing_steps'] += 1
    else:
        config['non_increasing_steps'] = 0


# def train_teacher(teacher_model, student_model, dataloader, device, config):
#     training_loss_history = []
#     val_loss_history = []
#     num_steps_to_achieve = []
#     teacher_updates = 0
#     non_increasing_steps = 0
#     student_updates = 0
#     best_loss_on_dev = config['max_loss']
#     # saved_log_probs = []
#
#     for i_episode in count(1):
#         input_pool = []
#         label_pool = []
#         count_sampled = 0
#         i_iter = 0
#
#         while i_iter < config['max_iter']:  # Don't infinite loop while learning
#             # for i_iter in range(self.config['max_iter']):
#             # collect training batch
#             for idx, (inputs, labels) in enumerate(dataloader['teacher_train_loader']):
#
#                 state_config = {
#                     'inputs': inputs.to(device),
#                     'labels': labels.to(device),
#                     'num_class': config['num_classes'],
#                     'i_iter': i_iter,
#                     'training_loss_history': training_loss_history,
#                     'best_loss_on_dev': best_loss_on_dev,
#                     'model': student_model,
#                     'dataloader': dataloader,
#                     'device': device
#                 }
#                 state = state_func(state_config)
#
#                 action_prb = teacher_model(state.detach())
#
#                 sampled_actions = torch.bernoulli(action_prb.data.squeeze())
#                 indices = torch.nonzero(sampled_actions)
#
#                 log_probs = []
#                 for idx, action in enumerate(sampled_actions):
#                     if action.data == 1:
#                         log_probs.append(torch.log(action_prb[idx]))
#                     else:
#                         log_probs.append(torch.log(1 - action_prb[idx]))
#                 teacher_model.saved_log_probs.append(torch.cat(log_probs).sum())
#
#                 if len(indices) == 0:
#                     continue
#
#                 count_sampled += len(indices)
#                 selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
#                 selected_labels = labels[indices.squeeze()].view(-1, 1)
#                 input_pool.append(selected_inputs)
#                 label_pool.append(selected_labels)
#
#                 i_iter += 1
#                 if count_sampled >= config['batch_size']:
#                     break
#
#             # generate data for train student model
#             inputs = torch.cat(input_pool, 0)[:config['batch_size']]
#             labels = torch.cat(label_pool, 0)[:config['batch_size']].squeeze()
#
#             input_pool_tmp, label_pool_tmp = [], []
#             input_pool_tmp.append(torch.cat(input_pool, 0)[config['batch_size']:])
#             label_pool_tmp.append(torch.cat(label_pool, 0)[config['batch_size']:])
#             input_pool = input_pool_tmp
#             label_pool = label_pool_tmp
#             count_sampled = input_pool[0].shape[0]
#
#             train_student_config = {
#                 'inputs': inputs,
#                 'labels': labels,
#                 'device': device,
#                 'model': student_model,
#                 'optimizer': optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
#
#             }
#             train_loss, train_acc = train_student(train_student_config)
#             training_loss_history.append(train_loss)
#             i_iter += 1
#             student_lr_scheduler(train_student_config['optimizer'], i_iter)
#
#             # val_student_config = {'dataloader': self.dataloader['dev_loader']}
#             val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
#             best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev
#
#             print('val loss: {}, val acc: {}'.format(val_loss, val_acc))
#             if val_acc > config['tau'] or i_iter == config['max_iter']:
#                 num_steps_to_achieve.append(i_iter)
#
#                 break
#
#         update_teacher_config = {
#             'i_iter': i_iter,
#             'max_iter': config['max_iter'],
#             'non_increasing_steps': non_increasing_steps,
#             'reward_T_histtory': teacher_model.reward_T_histtory,
#             'rewards_baseline': teacher_model.rewards_baseline,
#             'i_episode': i_episode,
#             'teacher_updates': teacher_updates,
#             'saved_log_probs': teacher_model.saved_log_probs,
#             'optimizer': optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0)
#
#         }
#         update_teacher(update_teacher_config)
#
#         teacher_model.reward_T_histtory
#         # reinit the student model
#         # self.student_model.apply(resnet._weights_init)
#
#         if non_increasing_steps >= config['max_non_increasing_steps']:
#             return


def select_action(state):
    action_prb = teacher_model(state.detach())
    m = Categorical(action_prb)
    action = m.sample()
    teacher_model.saved_log_probs.append(m.log_prob(action))
    return action


def train_l2t():
    num_steps_to_achieve = []
    non_increasing_steps = 0
    for i_episode in trange(300):
        training_loss_history = []
        best_loss_on_dev = config['max_loss']
        # val_loss_history = []
        teacher_updates = 0
        student_updates = 0
        i_iter = 0
        input_pool = []
        label_pool = []
        terminals = False
        count_sampled = 0
        storage = Storage(config['rollout_length'])

        # init the student
        student_model.apply(_weights_init)

        # one episode
        while True:
            # for i_iter in range(config['max_iter']):  # Don't infinite loop while learning
            # count_sampled = 0

            for idx, (inputs, labels) in enumerate(dataloader['teacher_train_loader']):

                # compute the state feature
                state_config = {
                    'inputs': inputs.to(device),
                    'labels': labels.to(device),
                    'num_class': config['num_classes'],
                    'student_iter': student_updates,
                    'training_loss_history': training_loss_history,
                    'best_loss_on_dev': best_loss_on_dev,
                    'model': student_model,
                    'max_loss': config['max_loss'],
                    'max_iter': config['max_iter'],
                    'device': device
                }
                state = state_func(state_config)

                # the teacher select action according to the state feature
                # action = select_action(state)
                prediction = teacher_model(state.detach())
                sampled_actions = prediction['a']

                # finish one step
                i_iter += 1

                # collect data to train student
                indices = torch.nonzero(sampled_actions)
                if len(indices) == 0:
                    continue

                count_sampled = len(indices)
                selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                selected_labels = labels[indices.squeeze()].view(-1, 1)
                input_pool.append(selected_inputs)
                label_pool.append(selected_labels)

                if count_sampled >= config['M']:
                    storage.add(prediction)

                    inputs = torch.cat(input_pool, 0)[:config['M']]
                    labels = torch.cat(label_pool, 0)[:config['M']].squeeze()

                    input_pool = []
                    label_pool = []

                    train_student_config = {
                        'inputs': inputs,
                        'labels': labels,
                        'model': student_model,
                        'device': device,
                        'optimizer': optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9,
                                               weight_decay=0.0001)
                    }

                    # train the student
                    train_loss, train_acc = train_student(train_student_config)
                    training_loss_history.append(train_loss)
                    student_updates += 1
                    student_lr_scheduler_cifar10(train_student_config['optimizer'], student_updates)


                    # val the student on validation set
                    val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
                    best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev

                    print(
                        '\nstudent_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                            student_updates, train_loss, train_acc, val_loss, val_acc))
                    writer.add_scalars('train_l2t/train', {'train_loss': train_loss, 'train_acc': train_acc},
                                       global_step=student_updates)
                    writer.add_scalars('train_l2t/val', {'val_loss': val_loss, 'val_acc': val_acc}, global_step=student_updates)

                    if student_updates % (config['rollout_length']+1) !=0:
                        terminals = val_acc >= config['tau'] or i_iter == config['max_iter']
                        storage.add({'r': val_acc.float().unsqueeze(-1),
                                     'm': torch.tensor(1 - terminals, device='cuda', dtype=torch.float).unsqueeze(-1)})

                    if student_updates % (config['rollout_length']+1) == 0:
                        storage.placeholder()
                        advantages = torch.tensor(np.zeros((config['batch_size'], 1)), device='cuda', dtype=torch.float)
                        returns = prediction['v'].detach()
                        for i in reversed(range(config['rollout_length'])):
                            returns = storage.r[i] + config['discount'] * storage.m[i] * returns
                            if not config['use_gae']:
                                advantages = returns - storage.v[i].detach()
                            else:
                                td_error = storage.r[i] + config['discount'] * storage.m[i] * storage.v[i + 1] - storage.v[i]
                                advantages = advantages * config['gae_tau'] * config['discount'] * storage.m[i] + td_error
                            storage.adv[i] = advantages.detach()
                            storage.ret[i] = returns.detach()

                        log_prob, value, returns, advantages, entropy = storage.cat(
                            ['log_pi_a', 'v', 'ret', 'adv', 'ent'])
                        policy_loss = -(log_prob * advantages).mean()
                        value_loss = 0.5 * (returns - value).pow(2).mean()
                        entropy_loss = entropy.mean()

                        teacher_optimizer.zero_grad()
                        (policy_loss - config['entropy_weight'] * entropy_loss + config['value_loss_weight'] * value_loss).backward(
                            retain_graph=True)
                        nn.utils.clip_grad_norm_(teacher_model.parameters(), config['gradient_clip'])
                        teacher_optimizer.step()
                        teacher_updates += 1
                        teacher_lr_scheduler(teacher_optimizer, teacher_updates)

                        print(
                            'Policy: Iterations [%d], policy_loss: %5.4f, value_loss: %5.4f, entropy_loss: %5.4f, rewards: %5.4f'
                            % (teacher_updates, policy_loss.cpu().item(), value_loss.cpu().item(),
                               entropy_loss.cpu().item(), returns.mean().cpu().item()))
                        writer.add_scalar('train_l2t/policy_loss', policy_loss, teacher_updates)
                        writer.add_scalar('train_l2t/value_loss', value_loss, teacher_updates)
                        writer.add_scalar('train_l2t/entropy_loss', entropy_loss, teacher_updates)
                        writer.add_scalar('train_l2t/return', returns.mean(), teacher_updates)

                        # reset the storage
                        storage.reset()
                        # num_steps_to_achieve.append(i_iter)
                        # reward_T = -math.log(i_iter / config['max_iter'])
                        # teacher_model.rewards.append(reward_T)
                        # teacher_model.reward_T_histtory.append(reward_T)
                        # done = True
                        # print('acc >= 0.93 at {}， reward_T: {}'.format(i_iter, reward_T))
                        # print('=' * 30)
                        # print(teacher_model.reward_T_histtory)
                        # print('=' * 30)



                if terminals == True:
                    break

            if terminals == True:
                break

        # update teacher
        # update_teacher_config = {
        #     'i_iter': i_iter,
        #     'max_iter': config['max_iter'],
        #     'non_increasing_steps': non_increasing_steps,
        #     'rewards': teacher_model.rewards,
        #     'rewards_baseline': teacher_model.rewards_baseline,
        #     'i_episode': i_episode,
        #     'reward_T_histtory': teacher_model.reward_T_histtory,
        #     'saved_log_probs': teacher_model.saved_log_probs,
        #     'optimizer': optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0)
        # }
        # update_teacher(update_teacher_config)
        #
        # writer.add_scalar('non_increasing_steps', non_increasing_steps)
        # teacher_model.rewards = []
        # teacher_model.saved_log_probs = []

        # if non_increasing_steps >= config['max_non_increasing_steps']:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(teacher_model.reward_T_histtory[-1], i_episode))
    return


def test_l2t():
    training_loss_history = []
    best_loss_on_dev = config['max_loss']
    # val_loss_history = []
    teacher_updates = 0
    student_updates = 0
    i_iter = 0
    input_pool = []
    label_pool = []
    terminals = False
    count_sampled = 0
    num_effective_data = 0
    test_acc_list = []
    # init the student
    student_model.apply(_weights_init)
    storage = Storage(config['rollout_length'])
    epochs = 100
    # one episode
    # while True:
    for epoch in trange(epochs):
        for idx, (inputs, labels) in enumerate(dataloader['student_train_loader']):

            # compute the state feature
            state_config = {
                'inputs': inputs.to(device),
                'labels': labels.to(device),
                'num_class': config['num_classes'],
                'student_iter': student_updates,
                'training_loss_history': training_loss_history,
                'best_loss_on_dev': best_loss_on_dev,
                'model': student_model,
                'max_loss': config['max_loss'],
                'max_iter': config['max_iter'],
                'device': device
            }
            state = state_func(state_config)

            # the teacher select action according to the state feature
            # action = select_action(state)
            prediction = teacher_model(state.detach())
            sampled_actions = prediction['a']

            # finish one step
            i_iter += 1

            # collect data to train student
            indices = torch.nonzero(sampled_actions)
            if len(indices) == 0:
                continue

            count_sampled = len(indices)
            selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
            selected_labels = labels[indices.squeeze()].view(-1, 1)
            input_pool.append(selected_inputs)
            label_pool.append(selected_labels)

            if count_sampled >= config['M']:
                storage.add(prediction)
                inputs = torch.cat(input_pool, 0)[:config['batch_size']]
                labels = torch.cat(label_pool, 0)[:config['batch_size']].squeeze()

                input_pool = []
                label_pool = []
                count_sampled = 0

                train_student_config = {
                    'inputs': inputs,
                    'labels': labels,
                    'model': student_model,
                    'device': device,
                    'optimizer': optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9,
                                           weight_decay=0.0001)
                }

                # train the student
                train_loss, train_acc = train_student(train_student_config)
                training_loss_history.append(train_loss)
                student_updates += 1
                num_effective_data += inputs.size(0)
                student_lr_scheduler_cifar10(train_student_config['optimizer'], student_updates)

                # val the student on validation set
                val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
                best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev

                # test on the test set
                test_loss, test_acc = val_student(student_model, dataloader['test_loader'], device)
                test_acc_list.append(test_acc)

                print(
                    'Test: student_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                        student_updates, train_loss, train_acc, val_loss, val_acc))
                print('Test: student_iter: {}, test loss: {:.4f}, test acc: {:.4f}'.format(student_updates, test_loss,
                                                                                           test_acc))
                writer.add_scalars('test_l2t/train', {'train_loss': train_loss, 'train_acc': train_acc},
                                   global_step=student_updates)
                writer.add_scalars('test_l2t/val', {'val_loss': val_loss, 'val_acc': val_acc}, global_step=student_updates)
                writer.add_scalars('test_l2t/test', {'test_loss': test_loss, 'test_acc': test_acc}, num_effective_data)
    return
    # if val_acc > config['tau'] or i_iter == config['max_iter']:
    #     num_steps_to_achieve.append(i_iter)
    #     reward_T = -math.log(i_iter / config['max_iter'])
    #     teacher_model.rewards.append(reward_T)
    #     teacher_model.reward_T_histtory.append(reward_T)
    #     done = True
    #     print('acc >= 0.93 at {}， reward_T: {}'.format(i_iter, reward_T))
    #     print('=' * 30)
    #     print(teacher_model.reward_T_histtory)
    #     print('=' * 30)
    #     writer.add_scalar('num_to_achieve', i_iter, i_episode)
    #     writer.add_scalar('reward', reward_T, i_episode)
    # else:
    #     teacher_model.rewards.append(0)

    #     if done == True:
    #         break
    #
    # if done == True:
    #     break

    # update teacher
    # update_teacher_config = {
    #     'i_iter': i_iter,
    #     'max_iter': config['max_iter'],
    #     'non_increasing_steps': non_increasing_steps,
    #     'rewards': teacher_model.rewards,
    #     'rewards_baseline': teacher_model.rewards_baseline,
    #     'i_episode': i_episode,
    #     'reward_T_histtory': teacher_model.reward_T_histtory,
    #     'saved_log_probs': teacher_model.saved_log_probs,
    #     'optimizer': optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0)
    # }
    # update_teacher(update_teacher_config)
    #
    # writer.add_scalar('non_increasing_steps', non_increasing_steps)
    # teacher_model.rewards = []
    # teacher_model.saved_log_probs = []
    #
    # if non_increasing_steps >= config['max_non_increasing_steps']:
    #     print("Solved! Running reward is now {} and "
    #           "the last episode runs to {} time steps!".format(teacher_model.reward_T_histtory[-1], i_episode))
    #     return


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
        'batch_size': 300,
        'max_non_increasing_steps': 10,
        'num_classes': 10,
        'max_loss': 2.47,
        'rollout_length': 5,
        'M': 128,
        'discount': 0.99,
        'use_gae': True,
        'gae_tau': 0.95,
        'entropy_weight': 0.01,
        'value_loss_weight': 1.0,
        'gradient_clip': 0.5
    }
    teacher_model = TeacherNetwork().to(device)
    student_model = resnet32().to(device)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.01, weight_decay=0)

    dataloader = data_loader_func(batch_sizes=config['batch_size'], path_to_dataset='./data')

    writer = SummaryWriter('./runs/a2c_cifar10')

    print('Training the teacher starts....................')
    start = time.time()
    train_l2t()
    time_train = time.time() - start

    model_save_dir = './result/a2c_cifar10/teacher'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(teacher_model.state_dict(), os.path.join(model_save_dir, 'teacher_step1.pth'))

    print('Done.\nTesting the teacher.....................')
    teacher_model.load_state_dict(torch.load(os.path.join(model_save_dir, 'teacher_step1.pth')))
    start = time.time()
    test_l2t()
    time_test = time.time() - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_train // 3600, time_train // 60, time_train % 60))
    print('Testing complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_test // 3600, time_test // 60, time_test % 60))

    model_save_dir = './result/a2c_cifar10/student'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(student_model.state_dict(), os.path.join(model_save_dir, 'student_step2.pth'))
