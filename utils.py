import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import argparse
from torch import nn
from torchvision.transforms import transforms
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
from backbones import backbone
from backbones import resnet12


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class MultiTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, image_size):
        self.transform = transform
        self.image_size = image_size

    def __call__(self, x):
        query_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        ])
        x = query_transform(x)
        return [transform(x) for transform in self.transform]

class OriandAugTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, aug_transform, aug_num=5):
        self.transform = transform
        self.aug_transform = aug_transform
        self.aug_num = aug_num

    def __call__(self, x):
        return [self.transform(x), [self.aug_transform(x) for i in range(self.aug_num)]]

def cos(feature, base_feature):
    scores = []
    for i in range(len(base_feature)):
        fea = base_feature[i]
        score = np.dot(feature, fea)
        scores.append(score)
    # print(scores)
    index = np.argsort(scores)
    select_feature = base_feature[index[0]]
    theta = scores[index[0]]
    # print(theta)
    theta = np.sqrt((1 + theta) / 2)
    interpolation = (1.0 / (2 * theta)) * feature + (1.0 / (2 * theta)) * select_feature
    # print(np.linalg.norm(feature))
    # print(np.linalg.norm(interpolation))
    return interpolation

def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    print(pretrained_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(model_dict.keys())
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    model.load_state_dict(model_dict)

    return model


def feature_transform(support_feature, base_feature):
    for i in range(len(support_feature)):
        # feature = max_similarity(support_feature[i], base_feature, 'cos')
        # if feature:
        #     index, select_feature = feature
        #     # support_feature[i] = support_feature[i]
        #     support_feature[i] = 0.8 * support_feature[i]
        #     for j in range(len(select_feature)):
        #         support_feature[i] += 0.2 * select_feature[j]
        # else:
        #     support_feature[i] = support_feature[i]
        index, similarity, select_feature = max_similarity(support_feature[i], base_feature, 'euclidean')
        support_feature[i] = 0.5 * support_feature[i]
        for j in range(len(select_feature)):
            support_feature[i] += 0.5 * select_feature[j]

    return support_feature


def max_similarity(support_feature, base_feature, measure):
    similarity = np.zeros([len(base_feature), ])
    for i in range(len(base_feature)):
        support_copy = support_feature.detach().cpu().numpy()
        base_copy = base_feature[i]
        if measure == 'cos':
            a_norm = np.linalg.norm(support_copy)
            b_norm = np.linalg.norm(base_copy)
            cos = np.dot(support_copy, base_copy) / (a_norm * b_norm)
            similarity[i] = cos

        if measure == 'euclidean':
            # support_copy /= np.linalg.norm(support_copy)
            # base_copy /= np.linalg.norm(base_copy)
            distance = -((support_copy - base_copy) ** 2).sum() / 64
            # print(distance)
            similarity[i] = distance

    k = 1
    # similarity = similarity[similarity > np.cos(45*np.pi/180)]
    # if len(similarity) == 0:
    #     return
    index = np.argsort(similarity)
    # print(index)
    print(similarity[index])
    index = index[len(base_feature) - k: len(base_feature)]
    return index, similarity[index], torch.from_numpy(base_feature[index]).cuda()


np.random.seed(0)
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

# def visualize(feature, label):
# model_dict = {
#     'Conv4': backbone.Conv4,
#     'Conv6': backbone.Conv6,
#     'ResNet10': backbone.ResNet10,
#     'ResNet18': backbone.ResNet18,
#     'ResNet34': backbone.ResNet34,
#     'ResNet12': resnet12.ResNet12}


if __name__ == '__main__':
    model_path = ''
    pretrained_dict = torch.load(model_path)
