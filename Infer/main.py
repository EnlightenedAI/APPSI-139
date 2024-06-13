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
sys.path.append('./TCSI_pp/Extraction_model/models')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from utils.Data_loading_only_test import dataload as dataload_rewrite


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
# def dataload(config, data_paths):
#     test_data = build_dataset(config, data_paths)
#     test_sencens = []
#     test_lens = []
#     test_mask = []
#     test_text = []
#     test_id = []
#     for i, test_s in enumerate(test_data):
#         test_sencens.append(test_s[0])
#         test_lens.append(test_s[1])
#         test_mask.append(test_s[2])
#         test_text.append(test_s[3])
#         test_id.append(i)
#     print(len(test_id))
#     test_dataset = TensorDataset(torch.tensor(test_sencens), torch.tensor(test_lens), torch.tensor(test_mask),
#                                  torch.tensor(test_id))
#     text_iter= DataLoader(test_dataset, batch_size=config.batch_size)
#     return text_iter,test_text

pl.seed_everything(42)
model_name =args.rewrite_model
tokenizer = MT5Tokenizer.from_pretrained(model_name)  # 'google/mt5-small
# tokenizer.save_pretrained("mt5")
t5_model = MT5ForConditionalGeneration.from_pretrained(model_name)  # 'google/mt5-small'
t5_config = MT5Config.from_pretrained(model_name)
t5_model=t5_model.to(device)
t5_model.config.eos_token_id = tokenizer.eos_token_id
t5_model.config.pad_token_id = tokenizer.pad_token_id

config = MT5Config.from_pretrained(model_name)
config.pad_size=150
config.tokenizer=tokenizer
# loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
# loss_fct=torch.nn.CrossEntropyLoss(ignore_index=-100)

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
        # print(encoder_outputs)
        if task=='important':
            # 二分类任务，使用编码器的输出
            # print(labels)
            # print(encoder_outputs.last_hidden_state.shape)
            classification_output = self.important(encoder_outputs.last_hidden_state[:,0,:])
            # print(classification_output.shape)
            return classification_output
        elif task=='risk':
            # 二分类任务，使用编码器的输出
            # print(labels)
            # print(encoder_outputs.last_hidden_state.shape)
            classification_output = self.risk(encoder_outputs.last_hidden_state[:,0,:])
            # print(classification_output.shape)
            return classification_output  
        elif task=='sensitive':
            # 二分类任务，使用编码器的输出
            # print(labels)
            # print(encoder_outputs.last_hidden_state.shape)
            classification_output = self.sensitive(encoder_outputs.last_hidden_state[:,0,:])
            # print(classification_output.shape)
            return classification_output  
        elif task=='multi':
            # 多分类任务，使用编码器的输出
            # print(encoder_outputs.last_hidden_state.shape)
            classification_output = self.multi(encoder_outputs.last_hidden_state[:,0,:])
            # print(classification_output.shape)
        
            
            if not labels is None:
                # loss = multi_loss(classification_output, labels)
                return loss,classification_output
            else:
                return classification_output

        elif task=='rewrite':
            # 生成任务，使用解码器
            # 确保 decoder_input_ids 是正确传递的
            # decoder_outputs = self.model.decoder(
            #     input_ids=decoder_input_ids,
            #     attention_mask=decoder_attention_mask,
            #     # past_key_values=encoder_outputs.past_key_values,
            #     encoder_hidden_states=encoder_outputs.last_hidden_state,
            #     encoder_attention_mask=attention_mask
            # )
            with autocast():
                output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                loss=output[0]
                print(output)
                print(labels)
            # sequence_output = decoder_outputs[0]
            # prediction_scores = self.model.lm_head(sequence_output)
            # # prediction_scores=prediction_scores.view(-1, self.config.vocab_size)
            # # print('预测：',prediction_scores.device,prediction_scores.shape)
            # loss = None
            # # print(labels)
            # # pdb.set_trace()
            # if labels is not None:
            #     loss = loss_fct(prediction_scores.view(-1, self.model.config.vocab_size), labels.view(-1))
            #     # print('loss：',loss.device,loss)
                
            return  loss
        elif task=='generate':
            # with torch.no_grad():
            output=self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=250,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )
            # output = self.model.decoder.generate(input_ids, encoder_outputs=encoder_outputs, max_length=50, num_beams=4, early_stopping=True)
                # print('一个一个又一个')
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
        # target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        # target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        # labels = copy.deepcopy(target_ids)
        # labels[labels == 0] = -100
        return {"source_ids": source_ids, "source_mask": src_mask}

    def _build(self):
        for inputs in self.true_false_adjective_tuples:
            input_sent = "summarization: " + inputs[:350]
            # ouput_sent = outputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True,return_tensors="pt"
            )
            # tokenized_targets = self.tokenizer.batch_encode_plus(
            #     [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True,return_tensors="pt"
            # )
            self.inputs.append(tokenized_inputs)
            # self.targets.append(tokenized_targets)

if __name__ == '__main__':
    model = CustomMT5Model(config,t5_model)
    model.load_state_dict(torch.load('mt5_mtl_model.ckpt'))

    #loading data
    print("loading data")
    data_path='Infer/test_data/test.txt'
    
    true_false_adjective_tuples_test=dataload_rewrite(data_path)

    # print(model)
    infer_dataset = FalseGenerationDataset(tokenizer,true_false_adjective_tuples_test, 150, 150)
    
    rewrite_val_dataloader = DataLoader(infer_dataset, batch_size=1)
    print('-----数据加载完毕-------')
    with torch.no_grad(): 

            iv_loss=0
            mv_loss=0
            riv_loss=0
            sv_loss=0
            #---------import dev-----------
            # pbar_bv = tqdm(total=len(important_test_iter), desc='important test {:2d}', leave=False)
            importants = np.array([[]], dtype=int)
            risks      = np.array([[]], dtype=int)
            sensitives  = np.array([[]], dtype=int)
            topics     = np.array([[]], dtype=int)
            rewrites=[]
            for batch in rewrite_val_dataloader:
                # print(batch)
                # pre=model(
                #     input_ids=batch['source_ids'].to(device),
                #     attention_mask=batch['source_mask'].to(device),
                #     # labels=batch[1].to(device),
                #     task='important'
                # )
                # predic = pre.data.argmax(axis=1)
                # print('1',predic)
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
            # print(len(infer_dataset),len(importants))

            print(importants)
            print(risks)
            print(sensitives)
            print(topics)
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
                
            # print(importants,risks,sensitives,topics)
            torch.cuda.empty_cache()
            



    # print(config1)
    # test_iter,test_text =dataload(config,data_paths)
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)

    #important Identification
    # important_model=step_1.Model(config1).to(config1.device)
    # config1.important_load_path ='./TCSI_pp_zh/models/CAPP_130_roberta_pretrain/important_Identification_roberta.ckpt'
    # print('loading important model')
    # important_model.load_state_dict(torch.load(config1.important_load_path))
    # important_model.eval()
    # print('run important model')
    # predict_all,pr_id=test(config1, important_model, test_iter)
    # with open('./TCSI_pp_zh/' + dataset +'/test_out_text.txt','w',encoding='utf-8') as f:
    #     for im_label,in_text in zip(predict_all,test_text):
    #         if im_label == 1:
    #             f.write(f'{in_text}'+'\n')

    # #Classification Identification
    # data_paths ='./TCSI_pp_zh/' + dataset +'/test_out_text.txt'
    # step_2_test_iter, step_2_test_text = dataload(config2, data_paths)
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)
    # Classification = step_2.Model(config2).to(config2.device)
    # config2.Classification_load_path = 'TCSI_pp_zh/models/CAPP_130_roberta_pretrain/multi_classification_roberta.ckpt'
    # print('loading Classification model')
    # Classification.load_state_dict(torch.load(config2.Classification_load_path))
    # Classification.eval()
    # print('run Classification model')
    # predict_all, pr_id = multi_test(config2, Classification, step_2_test_iter)
    # print(predict_all)

    # #Risk Identification
    # print('loading risk model')
    # risk_model = step_1.Model(config1).to(config1.device)  # x.Model构建模型，config.device=”cpu“
    # config1.risk_load_path = 'TCSI_pp_zh/models/CAPP_130_roberta_pretrain/risk_identification_roberta.ckpt'
    # risk_model.load_state_dict(torch.load(config1.risk_load_path))
    # risk_model.eval()
    # print('run risk model')
    # predict_risk_all, pr_id = test(config1, risk_model, step_2_test_iter)
    # print('id', pr_id)
    # print(len(pr_id), len(predict_all), len(test_text))
    # for risk_label, in_text in zip(predict_risk_all, step_2_test_text):
    #     if risk_label == 1:
    #         print('risk-sentence', in_text)
    # print(predict_risk_all)
    # with open('./TCSI_pp_zh/' + dataset +'/test_out_text.json', 'w', encoding='utf-8') as f:
    #     for label, in_text, risk_label in zip(predict_all, step_2_test_text, predict_risk_all):
    #         print(risk_label)
    #         f.write(json.dumps({'text': in_text, 'label': int(label), 'highlight': int(risk_label)},
    #                            ensure_ascii=False) + '\n')

    # #rewrite
    # data_name = './TCSI_pp_zh/' + dataset +'/test_out_text.json'
    # load_path = "./TCSI_pp_zh/models/CAPP_130_mt5_pretrain/mT5_small.ckpt"
    # true_false_adjective_tuples_test = dataload_rewrite(data_name)
    save_rewrite_path = 'Infer/test_data/test_rewrite_sentences.json'  # 保存结果
    # new_model = T5FineTuner().to(device)
    # new_model.save_pretrained("mt5")
    # new_model = new_model.load_from_checkpoint(checkpoint_path=load_path)
    # new_model.to(device)
    # new_model.eval()
    # new_model.model.to(device)
    # with open(save_rewrite_path, 'w', encoding='utf-8') as fp:
    #     for text in true_false_adjective_tuples_test:
    #         test_tokenized = tokenizer.encode_plus('summarization' + text[0][:350], return_tensors="pt")
    #         test_input_ids = test_tokenized["input_ids"].to(device)
    #         test_attention_mask = test_tokenized["attention_mask"].to(device)
    #         beam_outputs = new_model.model.generate(
    #             input_ids=test_input_ids,
    #             attention_mask=test_attention_mask,
    #             max_length=250,
    #             early_stopping=True,
    #             num_beams=10,
    #             num_return_sequences=1,
    #             no_repeat_ngram_size=2,
    #         )

    #         for beam_output in beam_outputs:
    #             sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #         sent = ''.join(sent.split())
    #         fp.write(json.dumps({'text': text[0], 'rewrite': sent, "label": text[1], 'highlight': text[2]},
    #                             ensure_ascii=False) + '\n')
    write_into_pp=''
    type_list=args.type_list
    type_title=['How We Collect Your Information',
                # '我们获取的应用权限',
                'How We Share Your Information',
                'How We Use Your Information',
                'How We Store Your Information',
                'Measures We Take to Protect Your Information',
                'How You Can Manage Your Information',
                'How to Contact Us',
                'Authorization and Changes to the Privacy Policy',
                'Terms for Special Groups',
                # '停止运营后我们会如何处理您的数据'
                ]
    
    # f=open(save_rewrite_path,'r',encoding='utf-8').readlines()
    # if type_list==[]:
    # type_list=[0,1,2,5]

    for i,type_i in enumerate(type_list):
        write_into_pp += f"<h3>{i+1}.{type_title[type_i]}</h3>\n"
        j=1
        for sen,important,topic,risk,sensitive,rewrite in zip(true_false_adjective_tuples_test,importants,topics,risks,sensitives,rewrites):
            # sen=json.loads(sen)
            # if not sen == 'un-important' :
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

    # Write summarition to HTML.
    with open(f'./Infer/test_data/pp_sum.html', "w") as file:
        file.write(html_content)
    print("Summarition has been written to HTML.")