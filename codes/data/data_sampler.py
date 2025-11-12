import math
import torch
from torch.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """
    # 以dataset的长度是485，ratio=10为例讲解。 首先生成一个从0到4849的随机排列，然后找到这0到4849的随机数对应0-485的值indices，返回迭代器，可以一个一个地访问indices
    def __init__(self, dataset, num_replicas, rank, ratio=1): #ratio 扩大数据集的比例
        self.dataset = dataset
        self.num_replicas = num_replicas # 参与训练的GPU数量，通常等于 world_size 也就是几台电脑
        self.rank = rank # 当前进程在所有进程中的排名，也就是当前电脑的名称
        self.epoch = 0 #当前 epoch，因为从0开始，默认为0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas) #例如，dataset的长度是485，ratio=10，扩充后样本数4850 每个进程需要采样的样本数，计算公式为
        self.total_size = self.num_samples * self.num_replicas # 只有一个电脑，没做分布式，total_size=4850

    def __iter__(self): # 返回一个迭代器，用于生成抽样的索引
        # deterministically shuffle based on epoch
        g = torch.Generator() # torch.Generator() 是一个生成器对象，用于生成随机数。它提供了一种可重复的随机数生成方式
        g.manual_seed(self.epoch) # 我们为生成器g设置了一个种子。这意味着每次epoch值相同的时候，生成的随机数序列都将是相同的
        indices = torch.randperm(self.total_size, generator=g).tolist() #生成一个从0到4849的随机排列。

        dataset_size = len(self.dataset) # 485
        indices = [v % dataset_size for v in indices]   # 找到0到4999的随机数对应0-499的值

        # subsample 这一步是分布式训练中的关键，它根据当前进程的 rank 和总进程数 self.num_replicas 对索引进行子采样。
        # 这样做确保了每个进程只处理数据集的一部分，且所有进程处理的数据不重叠
        indices = indices[self.rank:self.total_size:self.num_replicas] #从 indices 列表中，从索引 self.rank 开始，每隔 self.num_replicas 个元素取一个，直到 self.total_size
        assert len(indices) == self.num_samples # 4850

        return iter(indices) #返回 indices 列表的迭代器，这样数据加载器可以逐个访问这些索引，按索引加载数据

    def __len__(self):
        return self.num_samples  # 4850

    def set_epoch(self, epoch):
        self.epoch = epoch
