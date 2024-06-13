# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
# from transformers import AutoTokenizer,AutoModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import datetime
import os
class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'roberta_multi_hugging'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
            # dataset + '/../class1.txt').readlines()]                                # 类别名单
        # self.save_path = dataset + self.model_name + f'_{datetime.date.today()}.ckpt'        # 模型训练结果
        # self.save_path = dataset + self.model_name + f'_{datetime.date.today()}.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        print("-"*10,self.device,torch.cuda.device_count())
        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 9#len(self.class_list)                         # 类别数
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 32                                          # mini-batch大小
        self.pad_size = 150#150                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6                                      # 学习率
        self.bert_path = 'xlm-roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.hidden_size = 768
        self.focalloss_rate=0.1
        self.continue_train=False
        #4:6  4:96
        self.valstep=int(100)
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained("xlm-roberta-base")
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = []
        self.fc=nn.Linear(config.hidden_size, config.num_classes)
        # self.fc = nn.ModuleList([nn.Linear(config.hidden_size, config.num_classes) for _ in range(12)])
    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        # print()
        pooled = outputs['last_hidden_state'][:, 0, :]
        # 确保所有的操作都在同一个设备上
        # device = pooled.device  # 获取pooled所在的设备
        # print(device)
        # 将fc中的所有层移动到与pooled相同的设备
        # self.fc = [fc.to('cuda') for fc in self.fc]# 确保fc层也在正确的设备上

        # 现在所有的张量都在同一个设备上，可以安全地执行操作
        # out = self.fc[0](pooled)  # 假设self.fc[0]是第一个全连接层
        # print(out.device)
        # out = self.fc[0](pooled)
        # for i in range(1, len(self.fc)):
        #     out += self.fc[i](pooled)  # 累加其他全连接层的输出
            # print(i,out.device)
        # torch.mean(outputs['last_hidden_state'], dim=1)  # 从输出中获取 pooled representation
        # print(pooled.shape)
        out = self.fc(pooled)
        # print(out.shape)
        return out