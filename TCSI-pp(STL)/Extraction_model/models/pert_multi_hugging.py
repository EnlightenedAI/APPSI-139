import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
import datetime

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'pert_hugging'
        self.train_path = dataset + '/train.txt'
        self.dev_path = dataset + '/dev.txt'
        self.test_path = dataset + '/test.txt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("-"*10,self.device,torch.cuda.device_count())
        self.require_improvement = 10000
        self.num_classes = 9
        self.num_epochs = 30
        self.batch_size = 16
        self.pad_size = 150
        self.learning_rate = 5e-6
        self.bert_path ='hfl/english-pert-large'   
        self.continue_train=False
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/english-pert-base')
        self.hidden_size = 768
        self.focalloss_rate=0.4
        self.valstep=100
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained('hfl/english-pert-base')
        for param in self.bert.parameters():
            param.requires_grad = True
    
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(config.hidden_size, 512)
        self.fc15=nn.Linear(512,196)
        self.fc2 = nn.Linear(196, config.num_classes)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.conv1 = nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=8)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_conv = nn.Linear(128, config.num_classes)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_size=256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_lstm = nn.Linear(256 * 2, config.num_classes)
    def forward(self, x):
        context = x[0]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs['last_hidden_state'][:, 0, :]
        logits = self.fc(pooled)

        return logits