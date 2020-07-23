# learning_to_teach_pytroch
This is the implement of paper “Fan, Yang, et al. "Learning to teach." ICLR (2018).” in pytorch

For deatils please refer to [Learning to teach](https://openreview.net/pdf?id=HJewuJWCZ).

## Requirements
```bash
python == 3.6
pytorch >= 0.4.0
tensorboardX == 1.8
tqdm == 4.32.2 
```
## Dataset
```bash
mkdir data
cd data
```
Dataset directory is `data/`, so download the dataset(e.g. MNIST, CIFAR10) and move them to the `data/`

## Training and Test

- run Learning to Teach on MNIST
`python main_mnist.py`

- run Learning to Teach on CIFAR10
`python main_cifar10.py`

## No Teach
For details, please read the Experiment of the paper.

- run No Teach on MNIST
`python no_teach_mnist.py`

- run No Teach on CIFAR10
`python no_teach_cifar10.py`

```
@misc{
  sxzheng2019l2t,
  title={Learning to teach: A implement in PyTorch},
  author={Sixiao Zheng},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/sixiaozheng/learning_to_teach_pytroch}},
}
```
