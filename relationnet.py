import torch.nn as nn
import torch.nn.functional as F
import torch
from backbones.backbone import init_layer, Flatten


class RelationNet(nn.Module):
    def __init__(self, model_func, train_way, test_way, shot, hidden_size, relation='local'):
        super().__init__()
        self.model = model_func
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.feat_dim = self.model.final_feat_dim
        self.hidden_size = hidden_size
        # if not self.model.flatten:
        if relation == 'local':
            self.relation_module = RelationModule(self.feat_dim, self.hidden_size)
        if relation == 'global':
            self.relation_module = GlobalRelationModule(self.feat_dim)

    def forward(self, x):
        feature = self.model(x)
        return feature

    def set_forward(self, query, support):
        query_feature = self.forward(query)
        support_feature = self.forward(support)
        # print(support_feature.shape)
        if self.training:
            support_feature = support_feature.view(self.shot, self.train_way, *self.feat_dim).mean(dim=0)
        else:
            support_feature = support_feature.view(self.shot, self.test_way, *self.feat_dim).mean(dim=0)
        # support_feature = torch.sum(support_feature, 0).squeeze(0)
        # print(support_feature.shape)
        n = query_feature.shape[0]
        m = support_feature.shape[0]
        query_feature = query_feature.unsqueeze(0).repeat(m, 1, 1, 1, 1)
        support_feature = support_feature.unsqueeze(0).repeat(n, 1, 1, 1, 1)
        query_feature = torch.transpose(query_feature, 0, 1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((support_feature, query_feature), 2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs)
        return relations

    def set_forward_loss(self, query, support, label):
        relations = self.set_forward(query, support)
        if self.training:
            relations = relations.view(-1, self.train_way)
        else:
            relations = relations.view(-1, self.test_way)
        # print(relations.shape)
        # loss = F.mse_loss(relations, label)
        # pred = torch.argmax(relations, dim=1)
        # label = torch.argmax(label, dim=1)
        # acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

        loss = F.cross_entropy(relations, label)
        pred = torch.argmax(relations, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    # def distillation_forward_loss(self, query, support, label):
    #     query_feature = self.forward(query)
    #     origin_feature = self.forward(support).unsqueeze(0)
    #     support_rot = torch.rot90(support, 1, [2, 3])
    #     rot_feature = self.forward(support_rot).unsqueeze(0)
    #     for k in range(2, 4):
    #         rot_image = torch.rot90(support, k, [2, 3])
    #         feature = self.forward(rot_image).unsqueeze(0)
    #         rot_feature = torch.cat([rot_feature, feature], dim=0)
    #     support_feature = torch.cat([origin_feature, rot_feature], dim=0)
    #     # print(support_feature.shape)
    #     support_feature = support_feature.mean(0)
    #     rot_feature = rot_feature.mean(0)
    #     origin_feature = origin_feature.squeeze(0)
    #     if self.training:
    #         support_feature = support_feature.view(self.shot, self.train_way, *self.feat_dim).mean(dim=0)
    #     else:
    #         support_feature = support_feature.view(self.shot, self.test_way, *self.feat_dim).mean(dim=0)
    #     # support_feature = torch.sum(support_feature, 0).squeeze(0)
    #     # print(support_feature.shape)
    #     n = query_feature.shape[0]
    #     m = support_feature.shape[0]
    #     query_feature = query_feature.unsqueeze(0).repeat(m, 1, 1, 1, 1)
    #     support_feature = support_feature.unsqueeze(0).repeat(n, 1, 1, 1, 1)
    #     query_feature = torch.transpose(query_feature, 0, 1)
    #     extend_final_feat_dim = self.feat_dim.copy()
    #     extend_final_feat_dim[0] *= 2
    #     relation_pairs = torch.cat((support_feature, query_feature), 2).view(-1, *extend_final_feat_dim)
    #     relations = self.relation_module(relation_pairs)
    #     if self.training:
    #         relations = relations.view(-1, self.train_way)
    #     else:
    #         relations = relations.view(-1, self.test_way)
    #     loss = F.cross_entropy(relations, label)
    #     mse_loss = self.mseloss(origin_feature, rot_feature)
    #     loss = loss + mse_loss
    #     pred = torch.argmax(relations, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #
    #     return loss, acc

    def distillation_forward_loss(self, query, support, label):
        query_feature = self.forward(query)
        origin_feature = self.forward(support)
        rot_feature = [origin_feature]
        for k in range(1, 4):
            rot_image = torch.rot90(support, k, [2, 3])
            feature = self.forward(rot_image)
            rot_feature.append(feature)
        if self.training:
            for i in range(len(rot_feature)):
                rot_feature[i] = rot_feature[i].view(self.shot, self.train_way, *self.feat_dim).mean(dim=0)
        else:
            for j in range(len(rot_feature)):
                rot_feature[j] = rot_feature[j].view(self.shot, self.test_way, *self.feat_dim).mean(dim=0)
        # support_feature = torch.sum(support_feature, 0).squeeze(0)
        # print(support_feature.shape)
        n = query_feature.shape[0]
        m = origin_feature.shape[0]
        query_feature = query_feature.unsqueeze(0).repeat(m, 1, 1, 1, 1)
        for i in range(len(rot_feature)):
            rot_feature[i] = rot_feature[i].unsqueeze(0).repeat(n, 1, 1, 1, 1)
        query_feature = torch.transpose(query_feature, 0, 1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relations = []
        for i in range(len(rot_feature)):
            relation_pairs = torch.cat((rot_feature[i], query_feature), 2).view(-1, *extend_final_feat_dim)
            relation = self.relation_module(relation_pairs)
            relations.append(relation)
        if self.training:
            for i in range(len(relations)):
                relations[i] = relations[i].view(-1, self.train_way)
        else:
            for i in range(len(relations)):
                relations[i] = relations[i].view(-1, self.test_way)

        loss = F.cross_entropy(relations[0], label)
        rot_relation = relations[1] + relations[2] + relations[3] / 3
        rot_loss = F.cross_entropy(rot_relation, label)
        pred1 = F.softmax(relations[0], dim=1)
        pred2 = F.softmax(rot_relation, dim=1)
        distillation_loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
        loss = loss + rot_loss + distillation_loss

        pred = torch.argmax(relations[0], dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    def mutual_forward(self, logits_1, logits_2, label):
        logits_1 = logits_1.view(-1, self.train_way)
        logits_2 = logits_2.view(-1, self.train_way)
        pred1 = F.softmax(logits_1, dim=1)
        pred2 = F.softmax(logits_2, dim=1)
        loss = F.cross_entropy(logits_1, label)
        loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1)) + loss
        pred = torch.argmax(logits_1, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def defreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def mseloss(self, origin_feature, rot_feature):
        kernel_size = origin_feature.size(2)
        avg_pool = nn.AvgPool2d(kernel_size)
        origin_feature = avg_pool(origin_feature)
        rot_feature = avg_pool(rot_feature)
        origin_feature = origin_feature.view(origin_feature.size(0), -1)
        rot_feature = rot_feature.view(rot_feature.size(0), -1)
        loss = F.mse_loss(origin_feature, rot_feature)
        return loss

    # def pretrain_forward(self, x):
    #     feature = self.model(x)
    #     if self.fc:
    #         logits = self.fc(feature.squeeze(-1).squeeze(-1))
    #     else:
    #         raise ValueError("None pretrain setting")
    #     return logits
    #
    # def pretrain_forward_loss(self, x, label):
    #     logits = self.pretrain_forward(x)
    #     loss = F.cross_entropy(logits, label)
    #     pred = torch.argmax(logits, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #     return loss, acc
    #
    # def set_pretrain(self, num_classes):
    #     self.pretrain_classes = num_classes
    #     self.fc = nn.Linear(self.feat_dim, self.pretrain_classes)

    # def set_flatten(self, flatten):
    #     self.model.set_flatten(flatten)
    #     self.feat_dim = self.model.final_feat_dim
    #     if not flatten:
    #         self.relation_module = RelationModule(self.feat_dim, self.hidden_size).cuda()


class RelationModule(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_dim, hidden_size):
        super(RelationModule, self).__init__()
        padding = 1 if (input_dim[1] < 10) and (input_dim[2] < 10) else 0
        # self.layer1 = nn.Sequential(
        #                 nn.Conv2d(input_dim[0]*2,input_dim[0],kernel_size=3,padding=padding),
        #                 nn.BatchNorm2d(input_dim[0], momentum=1, affine=True),
        #                 nn.ReLU(),
        #                 nn.MaxPool2d(2))
        # self.layer2 = nn.Sequential(
        #                 nn.Conv2d(input_dim[0],input_dim[0],kernel_size=3,padding=padding),
        #                 nn.BatchNorm2d(input_dim[0], momentum=1, affine=True),
        #                 nn.ReLU(),
        #                 nn.MaxPool2d(2))
        self.layer1 = RelationConvBlock(input_dim[0] * 2, input_dim[0], padding=padding)
        self.layer2 = RelationConvBlock(input_dim[0], input_dim[0], padding=padding)

        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)
        self.fc1 = nn.Linear(input_dim[0] * shrink_s(input_dim[1]) * shrink_s(input_dim[2]), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = torch.sigmoid(self.fc2(out))
        out = self.fc2(out)
        return out


class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding=0):
        super(RelationConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class GlobalRelationModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(input_dim[1], stride=1)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(input_dim[0] * 2, input_dim[0] * 2 // 16)
        self.fc2 = nn.Linear(input_dim[0] * 2 // 16, input_dim[0] * 2 // 16 ** 2)
        self.fc3 = nn.Linear(input_dim[0] * 2 // 16 ** 2, 1)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.flatten(out)
        # print(out.shape)
        # out = self.fc1(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
