import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1) # 返回符合条件的元组的索引并转为一维向量
            ind = torch.from_numpy(ind) # 数组转换为张量
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls] # 采样n_cls个类
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per] # 采样n_per个样本
                batch.append(l[pos]) # 采样的n_cls个类中的n_per个样本的下标
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

