import argparse
import os.path as osp

from torchvision.models import ResNet
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader.mini_imagenet import MiniImageNet
from dataloader.cub import CUB
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
from backbones import backbone
from backbones import resnet12, wrn
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet
from utils import load_model

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12,
    'WRN28_10': wrn.WRN28_10}

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
    parser.add_argument('--model', default='protonet')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--step-size', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.5)
    # parser.add_argument('--aug', default='protonet')
    parser.add_argument('--save-epoch', type=int, default=30)
    parser.add_argument('--save-path', default='./miniP/')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.model == 'relationnet':
        if args.backbone == 'Conv4':
            backbone = backbone.Conv4NP()
        else:
            backbone = model_dict[args.backbone](flatten = False)
        model = RelationNet(backbone, args.train_way, args.test_way, args.shot, args.hidden_size, 'local')
    else:
        model = ProtoNet(model_dict[args.backbone](), args.train_way, args.test_way, args.shot, args.temperature)

    # print(model)
    if 'Conv' in args.backbone or 'ResNet12' in args.backbone or 'WRN' in args.backbone:
        image_size = 84
    else:
        image_size = 224

    if args.dataset == 'mini':
        trainset = MiniImageNet('train', image_size, True)
        valset = MiniImageNet('val', image_size, False)
    else:
        trainset = CUB('train', image_size, True)
        valset = CUB('val', image_size, False)
    # train_sampler = CategoriesSampler(trainset.label, 100,
    #                                   args.train_way, args.shot + args.query)
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=2, pin_memory=False)

    # val_sampler = CategoriesSampler(valset.label, 400,
    #                                 args.test_way, args.shot + args.query)
    val_sampler = CategoriesSampler(valset.label, 600,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=2, pin_memory=False)

    # if args.resume:
    #     path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shot' + '/'
    #     model_path = path + 'maxacc.pth'
    #     model.load_state_dict(torch.load(model_path))
    #     print("load model successfully")
    # else:
    #     path = 'pretrain/' + 'protonet' + args.backbone + '/'
    #     model_path = path + 'maxacc.pth'
    #     if os.path.exists(path):
    #         load_model(model, model_path)
    #         print("load pretrain model successfully")

    model.cuda()
    print(model)

    # print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9, nesterov=True,
    #                             weight_decay=0.0005, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # base_feature = np.load('base_feature.npy')
    # for j in range(len(base_feature)):
    #     norm = np.linalg.norm(base_feature[j])
    #     base_feature[j] = base_feature[j] / norm

    result = {}
    result['acc'] = 0
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        for i, batch in enumerate(train_loader):
            data, label_ori = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]
            # print(data_shot.shape)
            label = torch.arange(args.train_way).repeat(args.query) # 生成train_way个label,每个再重复query次
            label = label.type(torch.cuda.LongTensor)
            # if args.model == 'protonet':
            #     label = torch.arange(args.train_way).repeat(args.query)
            #     label = label.type(torch.cuda.LongTensor)
            # else:
            #     y = torch.from_numpy(np.tile(range(args.train_way), args.query))
            #     label = torch.zeros((len(y), args.train_way)).scatter_(1, y.unsqueeze(1), 1).cuda()

            # model.freeze()
            loss, acc = model.set_forward_loss(data_query, data_shot, label)
            # loss, acc = model.set_multi_forward(data_query, data_shot, label)
            # loss, acc = model.guide_forward_loss(data_query, data_shot, label, base_feature)
            # loss, acc = model.distillation_forward_loss(data_query, data_shot, label)
            # loss, acc = model.dynamic_process_feature(data_query, data_shot, label, epoch)
            # loss, acc = model.posteriori_forward(data_query, data_shot, label, base_feature)

            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))
            # print(count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # lr_scheduler.step()
        if epoch >= 450:
            model.eval()
            val = 0
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    data, _ = [_.cuda() for _ in batch]
                    p = args.shot * args.test_way
                    data_shot, data_query = data[:p], data[p:]
                    # label = torch.arange(args.test_way).repeat(args.query)
                    # label = label.type(torch.cuda.LongTensor)
                    label = torch.arange(args.test_way).repeat(args.query)
                    label = label.type(torch.cuda.LongTensor)
                    # if args.model == 'protonet':
                    #     label = torch.arange(args.test_way).repeat(args.query)
                    #     label = label.type(torch.cuda.LongTensor)
                    # else:
                    #     y = torch.from_numpy(np.tile(range(args.test_way), args.query))
                    #     label = torch.zeros((len(y), args.test_way)).scatter_(1, y.unsqueeze(1), 1).cuda()

                    loss, acc = model.set_forward_loss(data_query, data_shot, label)
                    # loss, acc = model.set_multi_forward(data_query, data_shot, label)
                    val += acc

            print("epoch {}, acc={:.4f}".format(epoch, val / len(val_loader)))

            if (val > result['acc']):
                result['acc'] = val
                path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(
                    args.shot) + 'shot' + '/'
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(), osp.join(path, 'maxacc' + '.pth'))

            if epoch % args.save_epoch == 0:
                path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(
                    args.shot) + 'shot' + '/'
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(), osp.join(path, str(epoch) + '.pth'))

        lr_scheduler.step()

    # np.save('base_feature', base_feature)
    # print(count)
