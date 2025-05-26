import torch
import torch.nn as nn
from transformers import AutoTokenizer, XLNetModel
import datetime

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'xlnet_hugging'
        self.train_path = dataset + '/train.txt'
        self.dev_path = dataset + '/dev.txt'
        self.test_path = dataset + '/test.txt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 600
        self.num_classes = 2
        self.num_epochs = 30
        self.batch_size = 64
        self.pad_size = 150
        self.learning_rate = 5e-6
        self.bert_path = "xlnet/xlnet-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.focalloss_rate=0.4
        self.valstep=int(100)
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = XLNetModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask)
        cls_hidden_state = outputs['last_hidden_state'][:, 0, :]
        pooled=cls_hidden_state
        out = self.fc(pooled)
        return out