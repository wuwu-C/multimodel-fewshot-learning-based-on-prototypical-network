import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from utils import feature_transform, max_similarity

class ProtoNet(nn.Module):
    def __init__(self, model_func, train_way, test_way, shot, temperature):
        super().__init__()
        self.model = model_func
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.pretrain_classes = None
        self.fc = None
        self.temperature = temperature
        # self.pretrain = None

    def forward(self, x):
        feature = self.model(x)
        print(feature.shape)
        return feature



    def set_forward(self, query, support):
        query_feature = self.forward(query)
        support_feature = self.forward(support)
        if self.training:
            support_feature = support_feature.reshape(self.shot, self.train_way, -1).mean(dim=0)
        else:
            support_feature = support_feature.reshape(self.shot, self.test_way, -1).mean(dim=0)

        # query_feature = self.power_transform(query_feature)
        # support_feature = self.power_transform(support_feature)

        return query_feature, support_feature

    def set_forward_loss(self, query, support, label):
        query_feature, support_feature = self.set_forward(query, support)
        logits = self.euclidean_metric(query_feature, support_feature)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    def set_forward_logits(self, query, support):
        query_feature, support_feature = self.set_forward(query, support)
        logits = self.euclidean_metric(query_feature, support_feature)
        return logits

    def set_multi_forward(self, query, support, label):
        query_feature, support_feature = self.set_forward(query, support)
        query_feature = query_feature.view(-1, 10, 64)
        support_feature = support_feature.view(-1, 10, 64)
        logits = self.multi_euclidean_metric(query_feature, support_feature)
        loss = 0
        for i in range(10):
            # print(logits[:, :, i].shape)
            loss += F.cross_entropy(logits[:, :, i], label)

        pred = torch.argmax(logits.sum(dim=2), dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc
            # print(logits.shape)

    def guide_forward_loss(self, query, support, label, base_feature):
        query_feature, support_feature = self.set_forward(query, support)

        similar_support = self.select_feature(support_feature.detach(), base_feature)
        similar_query = self.select_feature(query_feature.detach(), base_feature)
        # print(similar_support.shape)
        # print(similar_query.shape)

        logits = self.euclidean_metric(query_feature, support_feature)
        guide_logits = self.euclidean_metric(similar_query, similar_support)
        logits = logits.view(-1, self.train_way)
        guide_logits = guide_logits.view(-1, self.train_way)
        pred1 = F.softmax(logits, dim=1)
        pred2 = F.softmax(guide_logits, dim=1)
        # mse_loss = F.mse_loss(distance_1, distance_2)
        loss = F.cross_entropy(logits, label)
        kl_loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
        # print(kl_loss)
        loss = kl_loss + loss
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    def mutual_output(self, query, support):
        query_feature, support_feature = self.set_forward(query, support)
        logits = self.euclidean_metric(query_feature, support_feature)
        distances = self.cal_supportdis(support_feature)

        return logits, distances

    # def mutual_forward(self, logits_1, logits_2, label):
    #     logits_1 = logits_1.view(-1, self.train_way)
    #     logits_2 = logits_2.view(-1, self.train_way)
    #     pred1 = F.softmax(logits_1, dim=1)
    #     pred2 = F.softmax(logits_2, dim=1)
    #     loss = F.cross_entropy(logits_1, label)
    #     loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1)) + loss
    #     pred = torch.argmax(logits_1, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #     return loss, acc

    def mutual_forward(self, logits_1, logits_2, label):
        logits_1 = logits_1.view(-1, self.train_way)
        logits_2 = logits_2.view(-1, self.train_way)
        pred1 = F.softmax(logits_1, dim=1)
        pred2 = F.softmax(logits_2, dim=1)
        # mse_loss = F.mse_loss(distance_1, distance_2)
        loss = F.cross_entropy(logits_1, label)
        loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1)) + loss
        # loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1)) + loss + mse_loss
        pred = torch.argmax(logits_1, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    def posteriori_forward(self, query, support, label, base_feature):
        query_feature, support_feature = self.set_forward(query, support)
        support_feature = feature_transform(support_feature, base_feature)
        logits = self.euclidean_metric(query_feature, support_feature)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    def dynamic_process_feature(self, query, support, label, epoch):
        query_feature, support_feature = self.set_forward(query, support)
        # print(label_ori)
        # print(query_feature.shape, support_feature.shape, label.shape)
        if os.path.exists('memory_feature.npy'):
            base_feature = np.load('memory_feature.npy')
            # print(base_feature.shape)
            for i in range(len(support_feature)):
                temp_feature = support_feature[i].detach()
                # print(temp_feature.shape)
                index, similarity, select_feature = max_similarity(temp_feature, base_feature, 'euclidean')
                # index, similarity, select_feature = max_similarity(temp_feature, base_feature, 'cos')
                # print(index)
                if similarity >= -1.3:
                    support_feature[i] = 0.8 * support_feature[i]
                    for j in range(len(select_feature)):
                        support_feature[i] += 0.2 * select_feature[j]
                    base_feature[index] = 0.66 * base_feature[index] + 0.33 * temp_feature.cpu().numpy()
                else:
                    base_feature = np.append(base_feature, temp_feature.cpu().numpy()[np.newaxis,:], axis=0)

            print(base_feature.shape)
            np.save('memory_feature.npy', base_feature)

        logits = self.euclidean_metric(query_feature, support_feature)
        # print(logits)
        # print(logits.max(dim=1), label)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

        if epoch == 200 and not os.path.exists('memory_feature.npy'):
            np.save("memory_feature.npy", support_feature.detach().cpu().numpy())
        return loss, acc

    def predict(self, query, support, label):
        query_feature, support_feature = self.set_forward(query, support)
        # print(query_feature.shape)
        # print(support_feature.shape)
        # base_feature = np.load('base_pretrain_feature.npy')
        base_feature = np.load('memory_feature.npy')
        support_feature = feature_transform(support_feature, base_feature)
        logits = self.euclidean_metric(query_feature, support_feature)
        # loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return acc

    def pretrain_forward(self, x):
        feature = self.model(x)
        if self.fc:
            logits = self.fc(feature)
            # print(logits.shape)
        else:
            raise ValueError("None pretrain setting")
        return logits

    def pretrain_forward_loss(self, x, label):
        logits = self.pretrain_forward(x)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc

    def set_pretrain(self, num_classes):
        self.pretrain_classes = num_classes
        self.fc = nn.Linear(self.model.final_feat_dim, self.pretrain_classes)

    def set_flatten(self, flatten):
        self.model.set_flatten(flatten)

    def euclidean_metric(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)

        # logits = -((a - b) ** 2).sum(dim=2)
        logits = -((a - b) ** 2).sum(dim=2) / self.temperature    # copy from FEAT
        return logits

    def multi_euclidean_metric(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, 10, 64)
        b = b.unsqueeze(0).expand(n, m, 10, 64)

        # logits = -((a - b) ** 2).sum(dim=2)
        logits = -((a - b) ** 2).sum(dim=3) / self.temperature    # copy from FEAT
        return logits

    def cal_supportdis(self, support_feature):
        distances = None
        length = len(support_feature)
        for i in range(length):
            for j in range(i+1, length):
                distance = -((support_feature[i] - support_feature[j]) ** 2).sum(dim=0) / self.temperature
                distance = distance.unsqueeze(0)
                # print(distance.shape)
                if i == 0 and j == 1:
                    distances = distance
                else:
                    distances = torch.cat((distances, distance), dim=0)

        return distances

    def select_feature(self, feature, base_feature):
        output = None
        feature = feature.cpu().numpy()
        for i in range(len(feature)):
            norm = np.linalg.norm(feature[i])
            feature[i] = feature[i] / norm
            distances = []
            for j in range(len(base_feature)):
                distance = ((feature[i] - base_feature[j]) ** 2).sum()
                distances.append(distance)
            index = np.argmin(distances)
            if i == 0:
                output = torch.from_numpy(base_feature[index]).unsqueeze(0)
                # print(output.shape)
            else:
                output = torch.cat((output, torch.from_numpy(base_feature[index]).unsqueeze(0)), dim=0)
        return output.cuda()

    def power_transform(self, feature):
        delta = 0.5
        feature = torch.pow(feature + 1e-6, delta)
        # beta = 0.5
        # ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
        return feature