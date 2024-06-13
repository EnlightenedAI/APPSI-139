# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import AutoTokenizer,AutoModel
# from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import datetime

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        # self.model = BertModel.from_pretrained('hfl/english-pert-base')
        self.model_name = 'pert_hugging'
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
        self.num_classes = 2#len(self.class_list)                       # 类别数
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 150#150                                         # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6                                       # 学习率
        self.bert_path ='hfl/english-pert-large'   
        
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
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        # print()
        #原来的
        pooled = outputs['last_hidden_state'][:, 0, :]  # 从输出中获取 pooled representation
        # print(pooled.shape)
        logits = self.fc(pooled)
        # print(out.shape)

        #双层
        # cls_output = outputs.last_hidden_state[:, 0, :]
        # cls_output = self.dropout(cls_output)
        # x = self.fc1(cls_output)
        # x = self.relu(x)
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        # logits = self.fc2(x)

        #基于attention的
        # sequence_output = outputs.last_hidden_state
        # attn_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        # cls_output = attn_output[:, 0, :]
        # cls_output = self.dropout(cls_output)
        # logits = self.fc(cls_output)
        #基于cnn
        # sequence_output = outputs.last_hidden_state.permute(0, 2, 1)
        # conv_output = self.conv1(sequence_output)
        # conv_output = self.relu(conv_output)
        # pooled_output = self.pool(conv_output).squeeze(2)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.fc_conv(pooled_output)
        # #基于lstm
        # sequence_output = outputs.last_hidden_state
        # lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        # lstm_output = self.dropout(lstm_output)
        # logits = self.fc_lstm(lstm_output[:, -1, :])

        return logits