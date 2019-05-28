from __future__ import print_function
import warnings
import torch.utils.data as data
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import codecs


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

        teacher_train:29997
        student_train:29997
        dev:1495
        test:10000
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, transform=None, target_transform=None, split='train_teacher', seed=1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        # self.train = train  # training set or test set
        self.split = split
        self.seed = seed
        assert split in ['teacher_train', 'student_train', 'dev', 'test']

        if split in ['teacher_train', 'student_train', 'dev']:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        if split in ['teacher_train', 'student_train', 'dev']:
            self.data = self.data.numpy()
            self.targets = self.targets.numpy()

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

            np.random.seed(self.seed)

            for target, data in self.target_data_dict.items():
                data = np.vstack(data).reshape(-1, 28, 28)

                # data_idx = list(range(len(data)))
                num_data_half = data.shape[0] // 2
                num_data_dev = int(num_data_half * 0.05)
                # np.random.shuffle(data_idx)
                np.random.shuffle(data)

                # data_idx = np.arange(data.shape[0])
                # np.random.shuffle(data_idx)
                # data = data[data_idx]

                self.data_train_teacher.extend(data[:num_data_half])
                self.targets_train_teacher.extend([target] * num_data_half)
                self.data_train_student.extend(data[num_data_half:])
                self.targets_train_student.extend([target] * num_data_half)

                # np.random.shuffle(self.data_train_teacher)
                self.data_dev.extend(data[:num_data_dev])
                self.targets_dev.extend([target] * num_data_dev)

            data_idx = list(range(len(self.data_train_teacher)))
            np.random.shuffle(data_idx)
            self.data_train_teacher = np.vstack(self.data_train_teacher).reshape(-1, 28, 28)
            self.data_train_teacher = self.data_train_teacher[data_idx]
            self.targets_train_teacher = np.asarray(self.targets_train_teacher)[data_idx]

            np.random.shuffle(data_idx)
            self.data_train_student = np.vstack(self.data_train_student).reshape(-1, 28, 28)
            self.data_train_student = self.data_train_student[data_idx]
            self.targets_train_student = np.asarray(self.targets_train_student)[data_idx]

            data_idx = list(range(len(self.data_dev)))
            np.random.shuffle(data_idx)
            self.data_dev = np.vstack(self.data_dev).reshape(-1, 28, 28)
            self.data_dev = self.data_dev[data_idx]
            self.targets_dev = np.asarray(self.targets_dev)[data_idx]

            if split == 'teacher_train':
                self.data = torch.from_numpy(self.data_train_teacher)
                self.targets = torch.from_numpy(self.targets_train_teacher)
            elif split == 'student_train':
                self.data = torch.from_numpy(self.data_train_student)
                self.targets = torch.from_numpy(self.targets_train_student)
            elif split == 'dev':
                self.data = torch.from_numpy(self.data_dev)
                self.targets = torch.from_numpy(self.targets_dev)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

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


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def collate_fn(self, data):
    inputs, labels = zip(*data)
    # print (len(inputs), inputs[0].shape)
    labels = torch.LongTensor(labels)
    inputs = torch.cat([x.view(1, 1, 28, 28) for x in inputs], 0)
    return inputs, labels


if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt


    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        print(npimg.shape)
        # img = np.transpose(img, (1,2,0))
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.show()
        plt.pause(0.001)


    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    dataset = MNIST(root='../data', transform=transform, target_transform=None, split='student_train')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

    print(len(dataloader))
    images, labels = next(iter(dataloader))

    print(images.shape)
    print(' '.join('%5s' % labels[j] for j in range(50)))
    images_grid=torchvision.utils.make_grid(images)
    print(images_grid.shape)
    imshow(images_grid)

    class_img_list = [0]*10
    total = 0
    for images, labels in dataloader:
        for label in labels:
            class_img_list[label.item()]+=1
        total+=labels.size(0)

    print(class_img_list)
    print(total)



