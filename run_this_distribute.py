import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.distributed import fleet
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from paddle.io import Subset
import paddle.nn.functional as F
import paddle.distributed as dist

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

import numpy as np

import sys
import argparse
import importlib

import copy
import os
os.environ['FLAGS_check_nan_inf'] = '1'

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
paddle.set_device('gpu')

start_time = time.time()
strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)
fleet_init_time = time.time() - start_time

@paddle.no_grad()
class Metrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = paddle.zeros((self.n_classes, self.n_classes))
        self.reset()
    
    def reset(self):
        self.confusion_matrix = paddle.zeros((self.n_classes, self.n_classes))
        
    def update(self, predicts, labels):
        predicts = paddle.argmax(predicts, axis=1) # [B, C] -> [B]
        labels = labels.squeeze(1) # [B, 1] -> [B]
        for i in range(len(predicts)):
            self.confusion_matrix[int(labels[i]), int(predicts[i])] += 1
    
    def get_confusion_matrix(self):
        return self.confusion_matrix
    
    def get_accuracy(self, confusion_matrix):
        return paddle.trace(confusion_matrix) / paddle.sum(confusion_matrix)
    
    def get_precision(self, confusion_matrix):
        precision = paddle.zeros([self.n_classes])
        for i in range(self.n_classes):
            precision[i] = confusion_matrix[i, i] / paddle.sum(confusion_matrix[:, i])
        return precision
    
    def get_recall(self, confusion_matrix):
        recall = paddle.zeros([self.n_classes])
        for i in range(self.n_classes):
            recall[i] = confusion_matrix[i, i] / paddle.sum(confusion_matrix[i, :])
        return recall
    
    def get_f1(self, confusion_matrix):
        precision = self.get_precision(confusion_matrix)
        recall = self.get_recall(confusion_matrix)
        f1 = paddle.zeros([self.n_classes])
        for i in range(self.n_classes):
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        return f1
    
    
    def calculate(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix()
            
        accuracy = self.get_accuracy(confusion_matrix)
        precision = self.get_precision(confusion_matrix)
        recall = self.get_recall(confusion_matrix)
        f1 = self.get_f1(confusion_matrix)
        
        return {'accuracy': accuracy.numpy().item(), 'precision': paddle.mean(precision).numpy().item(), 'recall': paddle.mean(recall).numpy().item(), 'f1': paddle.mean(f1).numpy().item()}
    
def all_reduce(metrics):
    confusion_matrix = copy.deepcopy(metrics.get_confusion_matrix())
    dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
    metric_dict = metrics.calculate(confusion_matrix)
    return metric_dict

def train_epoch(model, train_loader, optim, loss_fn, metrics):
    start_time = time.time()
    rank = dist.get_rank()
    get_rank_time = time.time() - start_time
    
    model.train()
    loss_list = []
    
    acc_train_time = 0
    acc_reduce_time = 0
    acc_updatelog_time = 0
    acc_update_time = 0
    acc_back_time = 0
    
    if rank == 0:
        tbar = tqdm(train_loader())

    for data in tbar if rank == 0 else train_loader():
        start_time = time.time()
        x_data = data[0]            # 训练数据
        y_data = data[1].reshape((-1, 1))           # 训练数据标签
        predicts = model(x_data)    # 预测结果  
        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, y_data)
        acc_train_time += time.time() - start_time
        
        start_time = time.time()
        # 反向传播 
        loss.backward()
        acc_back_time += time.time() - start_time
        
        start_time = time.time()
        # 更新参数 
        optim.step()
        # 梯度清零
        optim.clear_grad()
        acc_update_time += time.time() - start_time
        
        start_time = time.time()
        # 计算准确率 等价于 prepare 中metrics的设置
        metrics.update(predicts, y_data)
        acc_updatelog_time += time.time() - start_time
        
        start_time = time.time()
        
        metric_dict = all_reduce(metrics)
        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / dist.get_world_size()
        
        acc_reduce_time += time.time() - start_time
        
        if rank == 0:
            tbar.set_postfix(loss=loss.numpy()[0], **metric_dict)
        
        start_time = time.time()
        loss_list.append(loss.numpy()[0])
        acc_updatelog_time += time.time() - start_time
    
    start_time = time.time()
    metric_dict = {}   
    metric_dict['loss'] = sum(loss_list) / len(loss_list)
    metric_dict.update(all_reduce(metrics))
    metrics.reset()
    acc_updatelog_time += time.time() - start_time
    
    return metric_dict, get_rank_time, acc_update_time, acc_train_time, acc_back_time, acc_reduce_time, acc_updatelog_time

@paddle.no_grad()
def val_epoch(model, val_dataloader, metrics):
    model.eval()
    with tqdm(val_dataloader()) as tbar:
        for data in tbar:
            x_data = data[0]            # 训练数据
            y_data = data[1].reshape((-1, 1))           # 训练数据标签        
            predicts = model(x_data)    # 预测结果  
    
            # 计算准确率
            metrics.update(predicts, y_data)
            metric_dict = all_reduce(metrics)
            
            tbar.set_postfix(**metric_dict)
            
    final_metrics = all_reduce(metrics)
    metrics.reset()
    return final_metrics

def parse_args():
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer (default: sgd)')
    parser.add_argument('--model', default='resnet18', type=str, help='model (default: resnet18)')
    parser.add_argument('--mode', default='single_machine', type=str, choices=['local', 'single_machine', 'multi_machines'], help='mode')
    
    return parser.parse_args()

if __name__ == '__main__':
    begin_time = time.time()
    
    args = parse_args()
    save_path = f'./save/{args.mode}/{current_time}_{args.model}_{args.optimizer}/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                to_rgb=True,
            ),
    ])

    start_time = time.time()
    Model = getattr(importlib.import_module('paddle.vision.models'), args.model)
    model = Model(pretrained=False, num_classes=10)
    model = fleet.distributed_model(model)
    init_model_time = time.time() - start_time
    
    start_time = time.time()
    # dataset与mnist的定义与使用高层API的内容一致
    dataset = Cifar10(mode='train', data_file='/public/data/image/cifar10/cifar-10-batches-py.tar.gz', transform=transform)
    idxs = np.random.permutation(len(dataset))
    train_dataset = Subset(dataset, idxs[:45000])
    val_dataset = Subset(dataset, idxs[45000:])
    test_dataset = Cifar10(mode='test', data_file='/public/data/image/cifar10/cifar-10-batches-py.tar.gz', transform=transform)

    # 构建分布式数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=200, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=200, num_workers=0)
    
    # 构建分布式数据采样器
    # 注意：需要保证 batch 中每个样本数据 shape 相同，若原尺寸不一，需进行预处理
    train_sampler = DistributedBatchSampler(train_dataset, args.batch_size, shuffle=True)
    val_sampler = DistributedBatchSampler(val_dataset, 200, shuffle=False)
    test_sampler = DistributedBatchSampler(test_dataset, 200, shuffle=False)

    # 构建分布式数据加载器
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=4)
    
    
    # 设置优化器
    Optim = getattr(importlib.import_module('paddle.optimizer'), args.optimizer)
    optim = Optim(parameters=model.parameters(), learning_rate=0.02)
    optim = fleet.distributed_optimizer(optim)
    # 设置损失函数
    loss_fn = F.cross_entropy
    
    dataset_optim_time = time.time() - start_time

    metrics = Metrics(n_classes=10)

    # 设置迭代次数
    epochs = 10
    
    metric_dict_list = {}
    period_train_time_list = []
    period_reduce_time_list = []
    period_updatelog_time_list = []
    period_val_time_list = []
    period_get_rank_time = []
    period_update_time_list = []
    period_back_time_list = []
    
    best_metric = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch}') 
        
        metric_dict, get_rank_time, acc_update_time, acc_train_time, acc_back_time, acc_reduce_time, acc_updatelog_time = train_epoch(model, train_loader, optim, loss_fn, metrics)
        
        period_get_rank_time.append(get_rank_time)
        period_update_time_list.append(acc_update_time)
        period_train_time_list.append(acc_train_time)
        period_reduce_time_list.append(acc_reduce_time)
        period_updatelog_time_list.append(acc_updatelog_time)
        period_back_time_list.append(acc_back_time)
        
        for k, v in metric_dict.items():
            if k not in metric_dict_list:
                metric_dict_list[k] = []
            metric_dict_list[k].append(v)
        
        start_time = time.time()
        val_metric = val_epoch(model, val_loader, metrics)
        if val_metric['accuracy'] > best_metric:
            best_metric = val_metric['accuracy']
            paddle.save(model.state_dict(), f'{save_path}/{args.model}_best.pdparams')
        duration = time.time() - start_time
        period_val_time_list.append(duration)
        
    total_time = int(time.time() - begin_time)
    
    for k, v in metric_dict_list.items():
        if k == 'loss':
            continue
        plt.plot(metric_dict_list['loss'], label='loss', marker='o')
        plt.plot(v, label=k, marker='^')
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(f'{save_path}/train_{k}.png')
    
    with open(f'{save_path}/metrics.txt', 'w') as f:
        f.write(f'Run Command: {sys.argv}\n\n')
        f.write(f'Init Fleet Time: {fleet_init_time:.4f} s\n')
        f.write(f'Init Model Time: {init_model_time:.4f} s\n')
        f.write(f'Dataset Optim Time: {dataset_optim_time:.4f} s\n')
        f.write(f'\nEpochs: {epochs}\n')
        f.write(f'Period Get Rank Time: {period_get_rank_time} s, Total time: {np.sum(period_get_rank_time)}, Mean time: {np.mean(period_get_rank_time)}\n\n')
        f.write(f'Period Train Time: {period_train_time_list} s, Total time: {np.sum(period_train_time_list)}, Mean time: {np.mean(period_train_time_list)}\n\n')
        f.write(f'Period Back Time: {period_back_time_list} s, Total time: {np.sum(period_back_time_list)}, Mean time: {np.mean(period_back_time_list)}\n\n')
        f.write(f'Period Update Grad Time: {period_update_time_list} s, Total time: {np.sum(period_update_time_list)}, Mean time: {np.mean(period_update_time_list)}\n\n')
        f.write(f'Period Reduce Time: {period_reduce_time_list} s, Total time: {np.sum(period_reduce_time_list)}, Mean time: {np.mean(period_reduce_time_list)}\n\n')
        f.write(f'Period Updatelog Time: {period_updatelog_time_list} s, Total time: {np.sum(period_updatelog_time_list)}, Mean time: {np.mean(period_updatelog_time_list)}\n\n')
        f.write(f'Period Val Time: {period_val_time_list} s, Total time: {np.sum(period_val_time_list)}, Mean time: {np.mean(period_val_time_list)}\n\n')
        
        f.write(f'Full Time: {total_time//60} min {total_time % 60}s\n')
        

