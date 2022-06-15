import argparse
import os.path as osp

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_model, OriandAugTransform

from dataloader.mini_imagenet import MiniImageNet
from dataloader.cub import CUB
from dataloader.Tieredimagenet import Tieredimagenet
from dataloader.FC100 import FC100
from dataloader.cifarFS import cifarFS
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
from backbones import backbone
from backbones import resnet12
from torchvision import transforms
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet
# from methods.basefinetine import BaseFinetine

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument('--backbone', default='Conv4')
    parser.add_argument('--dataset', default='mini')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--model', default='relationnet')
    parser.add_argument('--save-path', default='./miniP/')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    print(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if 'Conv' in args.backbone or 'ResNet12' in args.backbone:
        image_size = 84
    else:
        image_size = 224

    ori_transform = transforms.Compose([
        # transforms.Resize(84),
        transforms.Resize(92),
        # transforms.Resize(self.image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        # transforms.Resize(self.image_size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if args.dataset == 'mini':
        testset = MiniImageNet('test', image_size, False, OriandAugTransform(ori_transform, aug_transform))
        # testset = MiniImageNet('test', image_size, False)
        base_class = 64
    elif args.dataset == 'CUB':
        testset = CUB('test', image_size, False, OriandAugTransform(ori_transform, aug_transform))
        # testset = CUB('test', image_size, False)
        base_class = 50
    elif args.dataset == 'Tieredimagenet':
        testset = Tieredimagenet('test', image_size, False, OriandAugTransform(ori_transform, aug_transform))
        # testset = CUB('test', image_size, False)
        base_class = 351
    elif args.dataset == 'FC100':
        testset = FC100('test', image_size, False, OriandAugTransform(ori_transform, aug_transform))
        base_class = 60
    else:
        testset = cifarFS('test', image_size, False, OriandAugTransform(ori_transform, aug_transform))
        base_class = 64

    test_sampler = CategoriesSampler(testset.label, 2000,
                                    args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=2, pin_memory=True)

    if args.model == 'relationnet':
        if args.backbone == 'Conv4':
            backbone = backbone.Conv4NP()
        else:
            backbone = model_dict[args.backbone](flatten=False)
        model = RelationNet(backbone, args.train_way, args.test_way, args.shot, args.hidden_size, 'local')
    elif args.model == 'protonet':
        backbone = model_dict[args.backbone]()
        model = ProtoNet(backbone, args.train_way, args.test_way, args.shot, args.temperature)
    # else:
        # backbone = model_dict[args.backbone]()
        # model = BaseFinetine(backbone, args.train_way, args.test_way, args.shot)
    # trainset = MiniImageNet('train', image_size)
    # # train_sampler = CategoriesSampler(trainset.label, 100,
    # #                                   args.train_way, args.shot + args.query)
    # train_sampler = CategoriesSampler(trainset.label, 100,
    #                                   args.train_way, args.shot + args.query)
    # train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
    #                           num_workers=8, pin_memory=True)
    # val_sampler = CategoriesSampler(valset.label, 400,
    #                                 args.test_way, args.shot + args.query)
    # path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shot' + "model1" + '/'
    # path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shotmodel' +  '/'
    # model_path = path + 'maxacc.pth'
    # model.load_state_dict(torch.load(model_path))
    # print("load model successfully")
    # model.cuda()

    if args.pretrain:
        # path = 'CUB/' + args.model + args.backbone + '/'
        # path = args.save_path + args.model + args.backbone + '/'
        path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(
            args.shot) + 'shot' + '/'
        model_path = path + 'maxacc.pth'
        if os.path.exists(path):
            load_model(model, model_path)
            print("load pretrain model successfully")
    else:
        # path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shotmodel0' +  '/'
        # path = args.save_path + args.model + args.backbone + '/'
        path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(
            args.shot) + 'shot' + '/'
        model_path = path + 'maxacc.pth'
        #model.load_state_dict(torch.load(model_path))
        if os.path.exists(path):
            load_model(model, model_path)
            print("load model successfully")
    model.cuda()

    # for epoch in range(1, args.max_epoch + 1):
    model.eval()
    val = []
    # with torch.no_grad():
    # base_feature = np.load('base_interpolation_feature.npy')
    # ase_feature = np.load('cub_base_feature.npy')
    # base_feature = np.load('euc_select_base_feature.npy')
    # print(base_feature.shape)
    # base_feature_covar = np.load('contrastive_covar_matrix.npy')
    # all_base_feature = np.load('mini_base_all_feature.npy')
    # base_feature = np.mean(base_feature, axis=0)
    # base_feature = np.load('base_feature.npy')
    # base_feature = np.mean(base_feature, axis=0)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # data, labelori = [_.cuda() for _ in batch]
            # p = args.shot * args.test_way
            # data_shot, data_query = data[:p], data[p:]
            # # print(data_shot.shape)
            # label = torch.arange(args.test_way).repeat(args.query)
            # label = label.type(torch.cuda.LongTensor)

            data, labelori = [_ for _ in batch]
            p = args.shot * args.test_way
            ori_data = data[0].cuda()
            aug_data = data[1]
            aug_image_lists = []
            for aug in aug_data:
                aug_image_lists.append(aug[:p])

            data_shot, data_query = ori_data[:p], ori_data[p:]
            # print(data_query.shape)
            # print(data_shot.shape)
            label = torch.arange(args.test_way).repeat(args.query)
            label = label .type(torch.cuda.LongTensor)

            # support_feature = model(data_shot)
            # support_feature = support_feature.reshape(args.shot, args.test_way, -1).mean(dim=0)
            # visualize(support_feature, labelori[:args.test_way])

            if args.model == 'protonet' or args.model == 'relationnet':
                # acc = model.predict(data_query, data_shot, label)
                # loss, acc = model.set_multi_forward(data_query, data_shot, label)
                loss, acc = model.set_forward_loss(data_query, data_shot, label)
            else:
                # base_feature = np.load('base_feature.npy')
                # base_feature = np.mean(base_feature, axis=0)
                # acc = model.test_swim(data_query, data_shot, aug_image_lists, label, all_base_feature)
                #acc = model.evaluate_test_gen(data_query, data_shot, aug_image_lists, label, base_feature, 'SVM')
                # acc = model.contrastive_test_gen(data_query, data_shot, label, base_feature)
                #acc = model.evaluate1(data_query, data_shot, label)
                # acc = model.funetine_proto(data_query, data_shot, base_feature)
                acc = model.evaluate_aug1(data_query, data_shot, aug_image_lists, label)
                # acc = model.test_smote(data_query, data_shot, label)
                # acc = model.finetine_loop(data_query, data_shot, base_feature)

            val.append(acc)
            print(acc)

            # print(val)
        val = np.asarray(val)
        print("acc={:.4f} +- {:.4f}".format(np.mean(val), 1.96 * (np.std(val) / np.sqrt(len(val)))))





