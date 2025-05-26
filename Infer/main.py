# coding: UTF-8

import time
import copy
import numpy as np
from  utils.test_eval import train, init_network,test as test
from utils.text_train_multi_eval import test as multi_test
from importlib import import_module
import argparse
from utils.test_utils import build_dataset, build_iterator, get_time_dif
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch
import json
import os
import sys
sys.path.append('./TCSI-pp-V2(MTL)')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import pytorch_lightning as pl
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    MT5Config,
)
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--data', type=str, default='test_data', help='choose a Data')
parser.add_argument('--topic_list', type=list, default=[], help="choose a sublist from the list [0,1,2,3,4,5,6,7,8,9,10].")
args = parser.parse_args()
from utils.Data_loading_only_test import dataload as dataload_rewrite
pl.seed_everything(42)
model_name =args.rewrite_model
tokenizer = MT5Tokenizer.from_pretrained(model_name)  # 'google/mt5-small
t5_model = MT5ForConditionalGeneration.from_pretrained(model_name)  # 'google/mt5-small'
t5_config = MT5Config.from_pretrained(model_name)
t5_model=t5_model.to(device)
t5_model.config.eos_token_id = tokenizer.eos_token_id
t5_model.config.pad_token_id = tokenizer.pad_token_id

config = MT5Config.from_pretrained(model_name)
config.pad_size=150
config.tokenizer=tokenizer
class CustomMT5Model(torch.nn.Module):
    def __init__(self,config,model):
        super().__init__()
        self.important = torch.nn.Linear(config.hidden_size,2)
        self.risk = torch.nn.Linear(config.hidden_size,2)
        self.sensitive = torch.nn.Linear(config.hidden_size,2)
        self.multi = torch.nn.Linear(config.hidden_size,9)
        self.model=model
        self.config=config
    def forward(self, input_ids=None, attention_mask=None, labels=None, decoder_input_ids=None, decoder_attention_mask=None,encoder_outputs=None,task='multi'):
        # 编码器部分
        if not encoder_outputs:
            encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask)
        if task=='important':
            classification_output = self.important(encoder_outputs.last_hidden_state[:,0,:])
            return classification_output
        elif task=='risk':
            classification_output = self.risk(encoder_outputs.last_hidden_state[:,0,:])
            return classification_output  
        elif task=='sensitive':
            classification_output = self.sensitive(encoder_outputs.last_hidden_state[:,0,:])
            return classification_output  
        elif task=='multi':
            classification_output = self.multi(encoder_outputs.last_hidden_state[:,0,:])
            if not labels is None:
                return loss,classification_output
            else:
                return classification_output
        elif task=='rewrite':
            with autocast():
                output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                loss=output[0]
            return  loss
        elif task=='generate':
            output=self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=250,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )
            return output
class FalseGenerationDataset(Dataset):
    def __init__(self, tokenizer, tf_list, max_len_inp=150, max_len_out=150):
        self.true_false_adjective_tuples = tf_list
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer #token-model
        self.inputs = []
        self.templates=[]
        self.targets = []
        self.skippedcount = 0
        # self.model=model
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        return {"source_ids": source_ids, "source_mask": src_mask}

    def _build(self):
        for inputs in self.true_false_adjective_tuples:
            input_sent = "summarization: " + inputs[:350]
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True,return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)

if __name__ == '__main__':
    model = CustomMT5Model(config,t5_model)
    model.load_state_dict(torch.load('mt5_mtl_model.ckpt'))

    #loading data
    print("loading data")
    data_path='Infer/test_data/test.txt'
    
    true_false_adjective_tuples_test=dataload_rewrite(data_path)

    infer_dataset = FalseGenerationDataset(tokenizer,true_false_adjective_tuples_test, 150, 150)
    
    rewrite_val_dataloader = DataLoader(infer_dataset, batch_size=1)
    print('-----数据加载完毕-------')
    with torch.no_grad(): 

            iv_loss=0
            mv_loss=0
            riv_loss=0
            sv_loss=0
            #---------import dev-----------
            importants = np.array([[]], dtype=int)
            risks      = np.array([[]], dtype=int)
            sensitives  = np.array([[]], dtype=int)
            topics     = np.array([[]], dtype=int)
            rewrites=[]
            for batch in rewrite_val_dataloader:
                encoder_outputs=model.model.encoder(
                    input_ids=batch['source_ids'].to(device),
                    attention_mask=batch['source_mask'].to(device)
                    )
                pre=model(
                    encoder_outputs=encoder_outputs,
                    task='important'
                )
                important = pre.data.argmax(axis=1)
                
                pre=model(
                                    encoder_outputs=encoder_outputs,
                                    task='multi'
                                )
                
                topic = pre.data.argmax(axis=1)
                pre=model(
                    encoder_outputs=encoder_outputs,
                    task='risk'
                )
                risk = pre.data.argmax(axis=1)
                pre=model(
                    encoder_outputs=encoder_outputs,
                    task='sensitive'
                )
                sensitive = pre.data.argmax(axis=1)
                
                importants = np.append(importants, important.cpu().numpy())
                risks = np.append(risks, risk.cpu().numpy())
                sensitives = np.append(sensitives, sensitive.cpu().numpy())
                topics = np.append(topics, topic.cpu().numpy())
            for sen, important in zip(true_false_adjective_tuples_test,importants):
                if important==1:
                    sen=sen.replace('\n', '')
                    test_tokenized = tokenizer.encode_plus('summarization:'+sen, return_tensors="pt")
                    test_input_ids = test_tokenized["input_ids"].to(device)
                    test_attention_mask = test_tokenized["attention_mask"].to(device)
                    beam_outputs = model.model.generate(
                        input_ids=test_input_ids,
                        attention_mask=test_attention_mask,
                        max_length=250,
                        early_stopping=True,
                        num_beams=10,
                        num_return_sequences=1,
                        no_repeat_ngram_size=5,
                    )
                    for beam_output in beam_outputs:
                        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    rewrites.append(sent)
                else:
                    rewrites.append('un-important')
            torch.cuda.empty_cache()
    save_rewrite_path = 'Infer/test_data/test_rewrite_sentences.json'  # 保存结果
    write_into_pp=''
    type_list=args.type_list
    type_title=['How We Collect Your Information',
                'How We Share Your Information',
                'How We Use Your Information',
                'How We Store Your Information',
                'Measures We Take to Protect Your Information',
                'How You Can Manage Your Information',
                'How to Contact Us',
                'Authorization and Changes to the Privacy Policy',
                'Terms for Special Groups'
                ]

    for i,type_i in enumerate(type_list):
        write_into_pp += f"<h3>{i+1}.{type_title[type_i]}</h3>\n"
        j=1
        for sen,important,topic,risk,sensitive,rewrite in zip(true_false_adjective_tuples_test,importants,topics,risks,sensitives,rewrites):
            if important==1:
                if topic==int(type_i):
                    if risk==0 and sensitive==0:
                        write_into_pp += f"<p>（{j}）{rewrite}</p>\n"
                    elif risk==1:
                        write_into_pp += f'<div style="border-left: 5px solid #ff9900; padding: 1px; background-color: #fdfdfd;"><p><strong>Sensitive Warning</strong>:  （{j}）{rewrite}></p></div>\n'
                    elif sensitive==1:
                        write_into_pp += f'<div style="border-left: 5px solid #ff0000; padding: 1px; background-color: #fdfdfd;"><p><strong>Sensitive Warning</strong>:  （{j}）{rewrite}></p></div>\n'
                    else:
                        write_into_pp += f'<div style="border-left: 5px solid #00ff00; padding: 1px; background-color: #fdfdfd;"><p><strong>Sensitive Warning</strong>:  （{j}）{rewrite}></p></div>\n'
                    j+=1
        if j==1:
            write_into_pp += f'<p style="font-family: Arial, sans-serif;text-align: justify;">Nothing was found related to “{type_title[type_i]}“ </p>\n'

    html_content=f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>Privacy Policy Summarization</title>
    <style>
    body {{
      display: flex;
      flex-direction: column;
      height: 100vh;
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      text-align: justify;
    }}
    .container {{
        width: 90%;
        max-width: 1100px;
        margin: 0 auto;
        }}
    .centered-title {{
        text-align: left;
        }}
  </style>
      
    </head>
    <body>
    <div class="container">
    <h1>Privacy Policy Summarization</h1>
    {write_into_pp}
    </div>
    </body>
    </html>
    """

    with open(f'./Infer/test_data/pp_sum.html', "w") as file:
        file.write(html_content)
    print("Summarition has been written to HTML.")