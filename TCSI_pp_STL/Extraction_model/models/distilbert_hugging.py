# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
# from transformers import AutoTokenizer,AutoModel
# from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
import datetime

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'DistilBERT_hugging'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
            # dataset + '/../class1.txt').readlines()]                                # 类别名单
        # self.save_path = dataset + self.model_name + f'_{datetime.date.today()}.ckpt'        # 模型训练结果
        # self.save_path = dataset + self.model_name + f'_{datetime.date.today()}.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        print("-"*10,self.device,torch.cuda.device_count())
        self.require_improvement = 600                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2#len(self.class_list)                         # 类别数
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 64                                          # mini-batch大小
        self.pad_size = 150#150                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6                                     # 学习率
        self.bert_path = 'distilbert/distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.focalloss_rate=0.4
        self.valstep=int(100)
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = DistilBertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        # print()
        # pooled = torch.mean(outputs['last_hidden_state'], dim=1)  # 从输出中获取 pooled representation
        # print(pooled.shape)
        cls_hidden_state = outputs['last_hidden_state'][:, 0, :]  # 获取[CLS]标记的隐藏状态
        # pooled = self.dropout(cls_hidden_state)
        pooled=cls_hidden_state
        out = self.fc(pooled)
        # print(out.shape)
        return out