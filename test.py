import paddle
from paddle.io import DataLoader
from paddle.distributed import fleet
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from paddle.io import Subset

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

import numpy as np
import prettytable as pt

import sys
import argparse
import importlib

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
paddle.set_device('gpu')

class Metrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
    def update(self, predicts, labels):
        if not isinstance(predicts, np.ndarray):
            predicts = predicts.numpy()
        if not isinstance(labels, np.ndarray):
            labels = labels.numpy()
        predicts = np.argmax(predicts, axis=1)
        labels = labels.squeeze(1)
        for i in range(len(predicts)):
            self.confusion_matrix[labels[i]][predicts[i]] += 1
    
    def get_accuracy(self):
        return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
    
    def get_precision(self):
        precision = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            precision[i] = self.confusion_matrix[i][i] / np.sum(self.confusion_matrix[:, i])
        return precision
    
    def get_recall(self):
        recall = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            recall[i] = self.confusion_matrix[i][i] / np.sum(self.confusion_matrix[i, :])
        return recall
    
    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()
        f1 = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        return f1
    
    def calculate(self):
        accuracy = self.get_accuracy()
        precision = self.get_precision()
        recall = self.get_recall()
        f1 = self.get_f1()
        return {'accuracy': accuracy, 'precision': np.mean(precision), 'recall': np.mean(recall), 'f1': np.mean(f1)}
    
        

def run_epoch(model, train_loader, optim, loss_fn, metrics):
    model.train()
    loss_list = []
    with tqdm(train_loader()) as tbar:
        y_data_list = []
        preds_list = []
        for batch_id, data in enumerate(tbar):
            x_data = data[0]            # 训练数据
            y_data = data[1].reshape((-1, 1))           # 训练数据标签
            
            predicts = model(x_data)    # 预测结果  
            
            y_data_list.extend(y_data.numpy().flatten().tolist())
            preds_list.extend(predicts.numpy().argmax(axis=1).tolist())
            
            # 计算损失 等价于 prepare 中loss的设置
            loss = loss_fn(predicts, y_data)
                
            # 计算准确率 等价于 prepare 中metrics的设置
            metrics.update(predicts, y_data)
            metric_dict = metrics.calculate()
            tbar.set_postfix(loss=loss.numpy()[0], **metric_dict)
            
            # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中
            # 反向传播 
            loss.backward()
            # 更新参数 
            optim.step()
            # 梯度清零
            optim.clear_grad()
            
            loss_list.append(loss.numpy()[0])
    
    metric_dict = {}   
    metric_dict['loss'] = sum(loss_list) / len(loss_list)
    metric_dict.update(metrics.calculate())
    metrics.reset()
    
    return metric_dict


def val_epoch(model, val_dataloader, metrics):
    model.eval()
    with tqdm(val_dataloader()) as tbar:
        for data in tbar:
            x_data = data[0]            # 训练数据
            y_data = data[1].reshape((-1, 1))           # 训练数据标签        
            predicts = model(x_data)    # 预测结果  
    
            # 计算准确率
            metrics.update(predicts, y_data)
            metric_dict = metrics.calculate()
            
            tbar.set_postfix(**metric_dict)
            
    final_metrics = metrics.calculate()
    metrics.reset()
    
    return final_metrics
    

def parse_args():
    parser = argparse.ArgumentParser("cifar10")
    parser.add_argument('--model', default='resnet18', type=str, help='model (default: resnet18)')
    parser.add_argument('--save_path', required=True)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    save_path = args.save_path
    
    transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                to_rgb=True,
            ),
    ])

    Model = getattr(importlib.import_module('paddle.vision.models'), args.model)
    model = Model(pretrained=False, num_classes=10)
    
    test_dataset = Cifar10(mode='test', data_file='/public/data/image/cifar10/cifar-10-batches-py.tar.gz', transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=200, num_workers=4)
    
    # 将该模型及其所有子层设置为预测模式
    state_dict = paddle.load(f'{save_path}/{args.model}_best.pdparams')
    model.set_state_dict(state_dict)
    
    model.eval()
    test_metrics = Metrics(n_classes=10)
    for batch_id, data in enumerate(tqdm(test_loader())):
        x_data = data[0]            # 训练数据
        y_data = data[1].reshape((-1, 1))           # 训练数据标签
        predicts = model(x_data)    # 预测结果 
        test_metrics.update(predicts, y_data)
    
    test_result = test_metrics.calculate()
    table = pt.PrettyTable()
    table.field_names = ["Metric", "Value Per Class", "Macro Average"]
    table.add_row(["Accuracy", test_metrics.get_accuracy(), test_result['accuracy']])
    table.add_row(["Precision", test_metrics.get_precision(), test_result['precision']])
    table.add_row(["Recall", test_metrics.get_recall(), test_result['recall']])
    table.add_row(["F1", test_metrics.get_f1(), test_result['f1']])

    test_metrics.reset()
    print(table)
    
    with open(f'{save_path}/metrics.txt', 'a') as f:
        f.write(str(table)+'\n')



