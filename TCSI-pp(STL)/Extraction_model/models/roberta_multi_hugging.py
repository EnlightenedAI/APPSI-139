import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import datetime
import os
class Config(object):

    def __init__(self, dataset):
        self.model_name = 'roberta_multi_hugging'
        self.train_path = dataset + '/train.txt'
        self.dev_path = dataset + '/dev.txt'
        self.test_path = dataset + '/test.txt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("-"*10,self.device,torch.cuda.device_count())
        self.require_improvement = 10000
        self.num_classes = 9
        self.num_epochs = 30
        self.batch_size = 32
        self.pad_size = 150
        self.learning_rate = 5e-6
        self.bert_path = 'xlm-roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.hidden_size = 768
        self.focalloss_rate=0.1
        self.continue_train=False
        self.valstep=int(100)
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained("xlm-roberta-base")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc=nn.Linear(config.hidden_size, config.num_classes)
    def forward(self, x):
        context = x[0]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs['last_hidden_state'][:, 0, :]
        out = self.fc(pooled)
        return out