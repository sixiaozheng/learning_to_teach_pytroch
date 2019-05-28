from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

        teacher_train:25000
        student_train:25000
        dev:1250
        test:10000

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, transform=None, target_transform=None,
                 split='teacher_train', seed=1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        # self.train = train  # training set or test set
        self.split = split
        self.seed = seed
        assert split in ['train', 'teacher_train', 'student_train', 'dev', 'test']

        if split in ['train', 'teacher_train', 'student_train', 'dev']:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data)
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if split in ['teacher_train', 'student_train', 'dev']:
            self.data_train_teacher = []
            self.data_train_student = []
            self.data_dev = []
            self.data_test = []

            self.targets_train_teacher = []
            self.targets_train_student = []
            self.targets_dev = []
            self.targets_test = []

            self.target_data_dict = {}

            for data, target in zip(self.data, self.targets):
                if target not in self.target_data_dict.keys():
                    self.target_data_dict[target] = []
                self.target_data_dict[target].append(data)

            # data_tmp=[]
            # target_tmp = []
            # for target, data in self.target_data_dict.items():
            #     data=np.vstack(data)
            #     data=data.reshape(-1,32,32,3)
            #     data_tmp.extend(data)
            #     target_tmp.extend([target]*len(data))
            # data_tmp = np.vstack(data_tmp)
            # data_tmp = data_tmp.reshape(-1,32,32,3)
            #
            # self.data = data_tmp
            # self.targets = target_tmp

            np.random.seed(self.seed)

            for target, data in self.target_data_dict.items():
                data = np.vstack(data).reshape(-1, 32, 32, 3)

                num_data_half = data.shape[0] // 2
                # num_data_half = len(data) // 2
                num_data_dev = int(num_data_half * 0.05)

                data_idx = np.arange(data.shape[0])
                np.random.shuffle(data_idx)
                data = data[data_idx]

                self.data_train_teacher.extend(data[:num_data_half])
                self.targets_train_teacher.extend([target] * num_data_half)
                self.data_train_student.extend(data[num_data_half:])
                self.targets_train_student.extend([target] * num_data_half)
                self.data_dev.extend(data[:num_data_dev])
                self.targets_dev.extend([target] * num_data_dev)


            # if split == 'teacher_train':
            #     self.data = np.vstack(self.data_train_teacher).reshape(-1, 32, 32, 3)
            #     self.targets = self.targets_train_teacher
            # elif split == 'student_train':
            #     self.data = np.vstack(self.data_train_student).reshape(-1, 32, 32, 3)
            #     self.targets = self.targets_train_student
            # elif split == 'dev':
            #     self.data = np.vstack(self.data_dev).reshape(-1, 32, 32, 3)
            #     self.targets = self.targets_dev

            data_idx = list(range(len(self.data_train_teacher)))
            np.random.shuffle(data_idx)
            self.data_train_teacher = np.vstack(self.data_train_teacher).reshape(-1, 32, 32, 3)
            self.data_train_teacher = self.data_train_teacher[data_idx]
            self.targets_train_teacher = np.asarray(self.targets_train_teacher)[data_idx]

            np.random.shuffle(data_idx)
            self.data_train_student = np.vstack(self.data_train_student).reshape(-1, 32, 32, 3)
            self.data_train_student = self.data_train_student[data_idx]
            self.targets_train_student = np.asarray(self.targets_train_student)[data_idx]

            data_idx = list(range(len(self.data_dev)))
            np.random.shuffle(data_idx)
            self.data_dev = np.vstack(self.data_dev).reshape(-1, 32, 32, 3)
            self.data_dev = self.data_dev[data_idx]
            self.targets_dev = np.asarray(self.targets_dev)[data_idx]

            if split == 'teacher_train':
                self.data = self.data_train_teacher
                self.targets = self.targets_train_teacher.tolist()
            elif split == 'student_train':
                self.data = self.data_train_student
                self.targets = self.targets_train_student.tolist()
            elif split == 'dev':
                self.data = self.data_dev
                self.targets = self.targets_dev.tolist()

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                        ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    def imshow(img):
        img = img/2 +0.5
        npimg =img.numpy()
        # img = np.transpose(img, (1,2,0))
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.show()
        plt.pause(0.001)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                                    ])

    dataset = CIFAR10('../data/cifar-10/', transform=transform, split='teacher_train', seed=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    print(len(dataloader))

    images, labels = next(iter(dataloader))

    imshow(torchvision.utils.make_grid(images))
