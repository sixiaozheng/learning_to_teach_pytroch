import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


from network.teacher_model.teacher_mlp import teacher_mlp
from network.student_model.MLP import MLP, weights_init
from dataset.data_loader_mnist import data_loader_func
from utils.setseed import setup_seed
from scipy.stats import rankdata
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

    student_model.eval()
    outputs = student_model(inputs)
    average_train_loss = max_loss if len(state_config['training_loss_history']) == 0 else sum(
        state_config['training_loss_history']) / len(state_config['training_loss_history'])
    log_P_y = torch.log(outputs[range(n_samples), state_config['labels'].data]).reshape(-1, 1)

    mask = torch.ones(inputs.size(0), state_config['num_class']).to(state_config['device'])
    mask[range(n_samples), labels.data] = 0
    margin_value = (outputs[range(n_samples), labels.data] - torch.max(mask * outputs, 1)[0]).reshape(-1, 1)

    state_feactues = torch.zeros((inputs.size(0), 25)).to(device)
    state_feactues[range(n_samples), labels.data] = 1
    state_feactues[range(n_samples), 10] = state_config['student_iter'] / state_config['max_iter']
    print(average_train_loss)
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


def student_lr_scheduler_mnist(optimizer, iterations):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations % 50 == 0:
        lr = lr * 0.1
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
    rewards = teacher_model.rewards
    i_episode = config['i_episode']
    optimizer = config['optimizer']
    saved_log_probs = teacher_model.saved_log_probs
    reward_T_histtory = teacher_model.reward_T_histtory

    optimizer.zero_grad()

    reward_T = rewards[-1]
    teacher_model.rewards_baseline = teacher_model.rewards_baseline + (reward_T - teacher_model.rewards_baseline) / (
                i_episode + 1)
    baseline = 0.0 if i_episode == 0 else teacher_model.rewards_baseline
    policy_loss = -torch.cat(saved_log_probs).sum() * (reward_T - baseline)
    policy_loss = policy_loss / config['batch_size']
    policy_loss.backward()
    optimizer.step()

    writer.add_scalar('policy_loss', policy_loss, i_episode)
    writer.add_scalar('baseline', baseline, i_episode)

    teacher_lr_scheduler(optimizer, i_episode)

    last_reward = 0.0 if len(reward_T_histtory) == 1 else reward_T_histtory[-2]
    if abs(last_reward - reward_T) < 0.01:
        config['non_increasing_steps'] += 1
    else:
        config['non_increasing_steps'] = 0
    return config['non_increasing_steps']


def select_action(state):
    action_prb = teacher_model(state.detach())
    m = Categorical(action_prb)
    action = m.sample()
    teacher_model.saved_log_probs.append(m.log_prob(action))
    return action


def train_l2t():
    num_steps_to_achieve = []
    non_increasing_steps = 0
    for i_episode in trange(config['train_episode']):
        training_loss_history = []
        best_loss_on_dev = config['max_loss']
        student_updates = 0
        i_iter = 0
        input_pool = []
        label_pool = []
        done = False
        count_sampled = 0

        # init the student
        student_model.apply(weights_init)

        # one episode
        while True:

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
                action = select_action(state)

                # finish one step
                i_iter += 1

                # collect data to train student
                indices = torch.nonzero(action)
                if len(indices) == 0:
                    continue

                count_sampled += len(indices)
                selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                selected_labels = labels[indices.squeeze()].view(-1, 1)
                input_pool.append(selected_inputs)
                label_pool.append(selected_labels)

                if count_sampled >= config['batch_size']:
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
                        'optimizer': optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9,
                                               weight_decay=0.0001)
                    }

                    # train the student
                    train_loss, train_acc = train_student(train_student_config)
                    training_loss_history.append(train_loss)
                    student_updates += 1
                    student_lr_scheduler_mnist(train_student_config['optimizer'], student_updates)

                    # val the student on validation set
                    val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
                    best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev

                    print(
                        'episode:{}, student_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                            i_episode, student_updates, train_loss, train_acc, val_loss, val_acc))
                    # writer.add_scalars('train', {'train_loss': train_loss, 'train_acc': train_acc},
                    #                    global_step=student_updates)
                    # writer.add_scalars('val', {'val_loss': val_loss, 'val_acc': val_acc}, global_step=student_updates)

                    if val_acc >= config['tau'] or i_iter == config['max_iter']:
                        num_steps_to_achieve.append(i_iter)
                        reward_T = -math.log(i_iter / config['max_iter'])
                        teacher_model.rewards.append(reward_T)
                        teacher_model.reward_T_histtory.append(reward_T)
                        done = True
                        print('acc >= {} at {}， reward_T: {}'.format(config['tau'], i_iter, reward_T))
                        print('=' * 30)
                        print(teacher_model.reward_T_histtory)
                        print('=' * 30)
                        writer.add_scalar('num_to_achieve', i_iter, i_episode)
                        writer.add_scalar('reward', reward_T, i_episode)
                        break
                    else:
                        teacher_model.rewards.append(0)
            if done == True:
                break

        # update teacher
        update_teacher_config = {
            'non_increasing_steps': non_increasing_steps,
            'i_episode': i_episode,
            'optimizer': optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0),
            'batch_size': config['batch_size']
        }
        non_increasing_steps = update_teacher(update_teacher_config)

        writer.add_scalar('non_increasing_steps', non_increasing_steps)
        teacher_model.rewards = []
        teacher_model.saved_log_probs = []

        if non_increasing_steps >= config['max_non_increasing_steps']:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(teacher_model.reward_T_histtory[-1], i_episode))
            return


def test_l2t():
    training_loss_history = []
    best_loss_on_dev = config['max_loss']
    student_updates = 0
    i_iter = 0
    input_pool = []
    label_pool = []
    done = False
    count_sampled = 0
    num_effective_data = 0
    test_acc_list = []
    # init the student
    student_model.apply(weights_init)

    for epoch in trange(config['test_epoch']):
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
            action = select_action(state)

            # finish one step
            i_iter += 1

            # collect data to train student
            indices = torch.nonzero(action)
            if len(indices) == 0:
                continue

            count_sampled += len(indices)
            selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
            selected_labels = labels[indices.squeeze()].view(-1, 1)
            input_pool.append(selected_inputs)
            label_pool.append(selected_labels)

            if count_sampled >= config['batch_size']:
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
                    'optimizer': optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9,
                                           weight_decay=0.0001)
                }

                # train the student
                train_loss, train_acc = train_student(train_student_config)
                training_loss_history.append(train_loss)
                student_updates += 1
                num_effective_data += inputs.size(0)
                student_lr_scheduler_mnist(train_student_config['optimizer'], student_updates)

                # val the student on validation set
                val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
                best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev

                # test on the test set
                test_loss, test_acc = val_student(student_model, dataloader['test_loader'], device)
                test_acc_list.append(test_acc)

                print(
                    'Test: epoch: {}, student_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                        epoch, student_updates, train_loss, train_acc, val_loss, val_acc))
                print('Test: epoch{}, student_iter: {}, test loss: {:.4f}, test acc: {:.4f}'.format(epoch, student_updates, test_loss,
                                                                                           test_acc))
                writer.add_scalars('train', {'train_loss': train_loss, 'train_acc': train_acc},
                                   student_updates)
                writer.add_scalars('val', {'val_loss': val_loss, 'val_acc': val_acc}, student_updates)
                writer.add_scalars('test', {'test_loss': test_loss, 'test_acc': test_acc}, num_effective_data)
                if num_effective_data >= config['num_effective_data']:
                    return
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning to teach on mnist')
    parser.add_argument('--tau', type=float, default=0.93)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--train_episode', type=int, default=300)
    parser.add_argument('--test_epoch', type=int, default=150)
    parser.add_argument('--num_effective_data', type=int, default=1500000)
    args = parser.parse_args()

    # set seed
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'tau': args.tau,
        'max_iter': 10000,
        'batch_size': args.batch_size,
        'max_non_increasing_steps': 5,
        'num_classes': 10,
        'max_loss': 2.5,
        'train_episode': args.train_episode,
        'test_epoch': args.test_epoch,
        'num_effective_data': args.num_effective_data,
        'path_to_dataset': './data',
        'tensorboard_save_path': './runs/l2t_mnist',
        'teacher_save_dir':'./result/l2t/teacher',
        'teacher_save_model':'teacher_step1_mnist.pth',
        'student_save_dir':'./result/l2t/student',
        'student_save_model':'student_step2_mnist.pth'
    }
    teacher_model = teacher_mlp().to(device)
    student_model = MLP().to(device)

    dataloader = data_loader_func(batch_sizes=config['batch_size'], path_to_dataset=config['path_to_dataset'])

    writer = SummaryWriter(config['tensorboard_save_path'])

    print('Training the teacher starts....................')
    start = time.time()
    train_l2t()
    time_train = time.time() - start

    print('Saving the teahcer model........................')
    model_save_dir = config['teacher_save_dir']
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(teacher_model.state_dict(), os.path.join(model_save_dir, config['teacher_save_model']))

    print('Done.\nTesting the teacher.....................')
    print('Loading the teacher model')
    teacher_model.load_state_dict(torch.load(os.path.join(model_save_dir, config['teacher_save_model'])))
    start = time.time()
    test_l2t()
    time_test = time.time() - start

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_train // 3600, time_train // 60, time_train % 60))
    print('Testing complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_test // 3600, time_test // 60, time_test % 60))

    print('Saving the student mdoel.......................')
    model_save_dir = config['student_save_dir']
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(student_model.state_dict(), os.path.join(model_save_dir, config['student_save_model']))
