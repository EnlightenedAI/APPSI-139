from transformers import MT5Model,MT5ForConditionalGeneration, MT5TokenizerFast, MT5Config,ElectraConfig,GPT2Config
from torch.utils.data import DataLoader, Dataset
import torch
import time
from eval import rouge_scorces
import pdb
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from train_eval import train,test
from importlib import import_module
import argparse
from tqdm import tqdm
from utils import build_dataset, get_time_dif
from utils_multi import build_dataset_multi, get_time_dif
from torch.utils.data import TensorDataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torch.nn import DataParallel
import torch.distributed as dist
from sklearn.metrics import recall_score,f1_score,precision_score,roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, XLNetModel
import copy
import os
import json
from sklearn import metrics

classify =False
rewrite=False
nccl=False
test_step= False
train_step=False

test_rewrite=False
if test_rewrite:
    test_step= True
    rewrite=True

test_classify=True
if test_classify:
    classify=True
    test_step= True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=64,initial_growth_trigger_mb=512'
from Data_loading import dataload
from loss.focallooss import Focal_Loss,Focal_Loss_multi
from pytorch_pretrained.optimization import BertAdam
try:
    from torch.utils.data.distributed import DistributedSampler
except ImportError:
    raise ImportError("DistributedSampler required for distributed training.")
if nccl:
    dist.init_process_group(
            backend='nccl',  # 对于GPU训练，通常使用'nccl'
        )
binary_loss=Focal_Loss(alpha=0.4, gamma=2) 
multi_loss=Focal_Loss_multi(alpha=0.4, gamma=2) 
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=50256,reduction='mean')
from transformers import EncoderDecoderModel, BertTokenizer, GPT2Tokenizer
from transformers import ElectraTokenizer, ElectraModel
encoder_tokenizer = AutoTokenizer.from_pretrained('xlnet/xlnet-base-cased')
decoder_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
decoder_tokenizer.pad_token =decoder_tokenizer.eos_token
model = EncoderDecoderModel.from_encoder_decoder_pretrained('xlnet/xlnet-base-cased', 'openai-community/gpt2')
model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
model.config.pad_token_id = decoder_tokenizer.pad_token_id
config = AutoTokenizer.from_pretrained('xlnet/xlnet-base-cased')
dconfig=GPT2Config.from_pretrained('openai-community/gpt2')
print(config ,dconfig )
config.pad_size=150
config.hidden_size=768
config.decoder_vocab_size=dconfig.vocab_size
config.tokenizer=encoder_tokenizer
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
        if not encoder_outputs:
            encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask)
        if task=='important':
            classification_output = self.important(encoder_outputs.last_hidden_state[:,0,:])
            if not labels is None:
                loss = binary_loss(classification_output, labels)
                return loss,classification_output
            else:
                return classification_output
        elif task=='risk':
            classification_output = self.risk(encoder_outputs.last_hidden_state[:,0,:])
            if not labels is None:
                loss = binary_loss(classification_output, labels)
                return loss,classification_output
            else:
                return classification_output  
        elif task=='sensitive':
            classification_output = self.sensitive(encoder_outputs.last_hidden_state[:,0,:])
            if not labels is None:
                loss = binary_loss(classification_output, labels)
                return loss,classification_output
            else:
                return classification_output  
        elif task=='multi':
            classification_output = self.multi(encoder_outputs.last_hidden_state[:,0,:])
            if not labels is None:
                loss = multi_loss(classification_output, labels)
                return loss,classification_output
            else:
                return classification_output

        elif task=='rewrite':
                
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
model = CustomMT5Model(config,model)
if rewrite:
    max_len_inp=250
    max_len_out=150
    max_epochs=100
    train_is_ture=True
    testdata_is_ture=True
    lr=3e-4
    data_path=f'TCSI_pp/preprocessing/results_rewrite/'
    save_rewrite_path=f'./result/xlnet2gpt_rewrite_ddp_epoch{max_epochs}.json' #save rewrite result
    true_false_adjective_tuples_train,true_false_adjective_tuples_validation,true_false_adjective_tuples_test=dataload(data_path)


    class FalseGenerationDataset(Dataset):
        def __init__(self, encoder_tokenizer,decoder_tokenizer, tf_list, max_len_inp=150, max_len_out=150):
            self.true_false_adjective_tuples = tf_list
            self.max_len_input = max_len_inp
            self.max_len_output = max_len_out
            self.encoder_tokenizer = encoder_tokenizer #token-model
            self.decoder_tokenizer =decoder_tokenizer
            self.inputs = []
            self.templates=[]
            self.targets = []
            self.skippedcount = 0
            self._build()

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, index):
            source_ids = self.inputs[index]["input_ids"].squeeze()
            target_ids = self.targets[index]["input_ids"].squeeze()
            src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
            target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
            labels = copy.deepcopy(target_ids)
            labels[labels == 0] = -100
            return {"source_ids": source_ids, "source_mask": src_mask,  "target_ids": target_ids, "target_mask": target_mask,
                    "labels": labels}

        def _build(self):
            for inputs, outputs in self.true_false_adjective_tuples:
                input_sent = "summarization: " + inputs[:350]
                ouput_sent = outputs
                tokenized_inputs = self.encoder_tokenizer.batch_encode_plus(
                    [input_sent], max_length=self.max_len_input, pad_to_max_length=True,return_tensors="pt"
                )
                tokenized_targets = self.decoder_tokenizer.batch_encode_plus(
                    [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True,return_tensors="pt"
                )
                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)

    train_dataset = FalseGenerationDataset(encoder_tokenizer,decoder_tokenizer,true_false_adjective_tuples_train, max_len_inp, max_len_out)
    validation_dataset = FalseGenerationDataset(encoder_tokenizer,decoder_tokenizer,true_false_adjective_tuples_validation, max_len_inp, max_len_out)
    if nccl:
        sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        rewrite_train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
        sampler = DistributedSampler(validation_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        rewrite_val_dataloader = DataLoader(validation_dataset, batch_size=16, sampler=sampler)
    else:
        rewrite_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        rewrite_val_dataloader = DataLoader(validation_dataset, batch_size=1)

if classify:
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    start_time = time.time()
    print("Loading binary data...")
    def binary_dataload(config,data_paths,nccl,batch_size=8):
        train_data, dev_data, test_data = build_dataset(config,data_paths)
        train_sencens=[]
        train_lable=[]
        train_lens=[]
        train_mask=[]
        z=0
        o=0
        for train_s in train_data:
            if train_s[1]==1:
                o+=1
            else:
                z+=1
            train_sencens.append(train_s[0])
            train_lable.append(train_s[1])
            train_lens.append(train_s[2])
            train_mask.append(train_s[3])
        print(z,o,z/(z+o))
        train_dataset = TensorDataset(torch.tensor(train_sencens),torch.tensor(train_lable),torch.tensor(train_lens),torch.tensor(train_mask))
        test_sencens=[]
        test_lable=[]
        test_lens=[]
        test_mask=[]
        z=0
        o=0
        for test_s in test_data:
            if test_s[1]==1:
                o+=1
            else:
                z+=1
            test_sencens.append(test_s[0])
            test_lable.append(test_s[1])
            test_lens.append(test_s[2])
            test_mask.append(test_s[3])
        print(z,o,z/(z+o))
        test_dataset = TensorDataset(torch.tensor(test_sencens), torch.tensor(test_lable), torch.tensor(test_lens),
                                        torch.tensor(test_mask))
        dev_sencens = []
        dev_lable = []
        dev_lens = []
        dev_mask = []
        z=0
        o=0
        for dev_s in dev_data:
            if dev_s[1]==1:
                o+=1
            else:
                z+=1
            dev_sencens.append(dev_s[0])
            dev_lable.append(dev_s[1])
            dev_lens.append(dev_s[2])
            dev_mask.append(dev_s[3])
        print(z,o,z/(z+o))
        dev_dataset = TensorDataset(torch.tensor(dev_sencens), torch.tensor(dev_lable), torch.tensor(dev_lens),
                                        torch.tensor(dev_mask))

        if nccl:
            sampler_b = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            binary_train_iter = DataLoader(train_dataset, batch_size=12, sampler=sampler_b)
            sampler_b = DistributedSampler(dev_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            binary_dev_iter = DataLoader(dev_dataset, batch_size=12, sampler=sampler_b)
            sampler_b = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            binary_test_iter = DataLoader(test_dataset, batch_size=12, sampler=sampler_b)
        else:
            binary_train_iter = DataLoader(train_dataset,sampler=ImbalancedDatasetSampler(train_dataset), batch_size=batch_size)
            binary_dev_iter = DataLoader(dev_dataset, batch_size=batch_size)
            binary_test_iter = DataLoader(test_dataset, batch_size=batch_size)
            return binary_train_iter,binary_dev_iter,binary_test_iter
    data_paths="TCSI_pp/preprocessing/results_new10/class12"
    data_paths_risk="TCSI_pp/preprocessing/results_new10/class13"
    data_paths_sensitive="TCSI_pp/preprocessing/results_new10/class14"
    important_train_iter,important_dev_iter,important_test_iter=binary_dataload(config,data_paths,nccl,batch_size=128)
    risk_train_iter,risk_dev_iter,risk_test_iter=binary_dataload(config,data_paths_risk,nccl,batch_size=64)
    sensitive_train_iter,sensitive_dev_iter,sensitive_test_iter=binary_dataload(config,data_paths_sensitive,nccl,batch_size=64)
    data_multi_paths='TCSI_pp/preprocessing/results_new10/class_multi_a'
    def multi_dataload(config,data_paths,nccl,batchsize=16):
        start_time = time.time()
        print("Loading multi data...")
        train_data, dev_data, test_data = build_dataset_multi(config,data_paths)
        train_sencens=[]
        train_lable=[]
        train_lens=[]
        train_mask=[]
        for train_s in train_data:
            train_sencens.append(train_s[0])
            train_lable.append(train_s[1])
            train_lens.append(train_s[2])
            train_mask.append(train_s[3])
        train_dataset = TensorDataset(torch.tensor(train_sencens),torch.tensor(train_lable),torch.tensor(train_lens),torch.tensor(train_mask))
        test_sencens=[]
        test_lable=[]
        test_lens=[]
        test_mask=[]
        for test_s in test_data:
            test_sencens.append(test_s[0])
            test_lable.append(test_s[1])
            test_lens.append(test_s[2])
            test_mask.append(test_s[3])
        test_dataset = TensorDataset(torch.tensor(test_sencens), torch.tensor(test_lable), torch.tensor(test_lens),
                                        torch.tensor(test_mask))
        dev_sencens = []
        dev_lable = []
        dev_lens = []
        dev_mask = []
        for dev_s in dev_data:
            dev_sencens.append(dev_s[0])
            dev_lable.append(dev_s[1])
            dev_lens.append(dev_s[2])
            dev_mask.append(dev_s[3])
        dev_dataset = TensorDataset(torch.tensor(dev_sencens), torch.tensor(dev_lable), torch.tensor(dev_lens),
                                        torch.tensor(dev_mask))
        if nccl:
            sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            multi_train_iter = DataLoader(train_dataset, batch_size=12, sampler=sampler)
            sampler = DistributedSampler(dev_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            multi_dev_iter = DataLoader(dev_dataset, batch_size=12, sampler=sampler)
            sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            multi_test_iter = DataLoader(test_dataset, batch_size=12, sampler=sampler)
        else:
            multi_train_iter = DataLoader(train_dataset, shuffle=True,batch_size=batchsize)

            multi_dev_iter = DataLoader(dev_dataset, batch_size=batchsize)
            multi_test_iter = DataLoader(test_dataset, batch_size=batchsize)
        return multi_train_iter,multi_dev_iter,multi_test_iter

    multi_train_iter,multi_dev_iter,multi_test_iter=multi_dataload(config,data_multi_paths,nccl,batchsize=64)
if nccl:
    device = torch.device("cuda", dist.get_rank())
    model.to(device)
    model = DDP(model,find_unused_parameters=True)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.torch.cuda.device_count() > 1:
        model = DataParallel(model)
    
print('-'*10,device,torch.cuda.device_count())

param_optimizer = list(model.named_parameters())
if rewrite:
    decoder_params = [p for n, p in param_optimizer if 'decoder' in n]
    optimizer_rw = torch.optim.AdamW(decoder_params, lr=lr, weight_decay=1e-5)
if classify:
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    num_epochs=100
    optimizer_r = BertAdam(optimizer_grouped_parameters,
                            lr=5e-6,
                            warmup=0.05,
                            t_total=(len(important_train_iter)+len(risk_train_iter)+len(sensitive_train_iter)+len(multi_train_iter)) * num_epochs)
if train_step:
    model.train()
    num_epochs=100
    max_dev_f1=0
    min_dev_loss=float('inf')
    if classify:
        for epoch in tqdm(range(30), desc="Epochs"):  # 例如，训练3个epoch
            i_loss=0
            m_loss=0
            risk_loss=0
            s_loss=0
            pbar_b = tqdm(total=len(important_train_iter), desc='important train batch {:2d}'.format(epoch), leave=False)
            predict_all = np.array([[]], dtype=int)
            labels_all = np.array([[]], dtype=int)
            for batch in important_train_iter:
                model.zero_grad()
                loss,pre=model(
                    input_ids=batch[0].to(device),
                    attention_mask=batch[3].to(device),
                    labels=batch[1].to(device),
                    task='important'
                )
                loss = torch.mean(loss)
                i_loss+=loss
                loss.backward()
                optimizer_r.step()
                pbar_b.update(1)
                pbar_b.set_postfix(loss='{:.3f}'.format(loss.item()))
                predic = pre.data.argmax(axis=1)
                labels_all = np.append(labels_all, batch[1].cpu().numpy())
                predict_all = np.append(predict_all, predic.cpu().numpy())
            pbar_b.close()
            torch.cuda.empty_cache()
            f1 = f1_score(labels_all, predict_all,average=None)
            print("----important train f1----:",f1)
            micro_important_f1 = f1_score(labels_all, predict_all, average='micro')
            macro_important_f1 = f1_score(labels_all, predict_all, average='macro')
            print(f"important train Micro-F1: {micro_important_f1}")
            print(f"important train Macro-F1: {macro_important_f1}")
            print('important train_loss:',i_loss/len(important_train_iter))
            pbar_m = tqdm(total=len(multi_train_iter), desc='multi batch {:2d}'.format(epoch), leave=False)
            predict_all = np.array([[]], dtype=int)
            labels_all = np.array([[]], dtype=int)
            for batch in multi_train_iter:
                model.zero_grad()
                loss,pre=model(
                    input_ids=batch[0].to(device),
                    attention_mask=batch[3].to(device),
                    labels=batch[1].to(device),
                    task='multi'
                )
                loss = torch.mean(loss)
                m_loss+=loss
                loss.backward()
                optimizer_r.step()
                pbar_m.update(1)
                pbar_m.set_postfix(loss='{:.3f}'.format(loss.item()))
                loss = torch.mean(loss)
                m=torch.nn.Sigmoid()
                predic = m(pre.data)
                Threshold = 0.5
                predic[predic > Threshold] = 1
                predic[predic <= Threshold] = 0
                if labels_all.shape[1] == 0:
                    labels_all=batch[1].cpu().numpy()
                    predict_all=predic.cpu().numpy()
                else:
                    labels=batch[1].cpu().numpy()
                    predic = predic.cpu().numpy()
                    labels_all = np.vstack((labels_all, labels))
                    predict_all = np.vstack((predict_all, predic))
            f1 = f1_score(labels_all, predict_all,average=None)
            print("----multi train f1----:",f1)
            micro_f1 = f1_score(labels_all, predict_all, average='micro')
            macro_multi_f1 = f1_score(labels_all, predict_all, average='macro')
            
            print(f"multi train Micro-F1: {micro_f1}")
            print(f"multi train Macro-F1: {macro_multi_f1}")
            pbar_m.close()
            torch.cuda.empty_cache()
            print('multi train_loss:',m_loss/len(multi_train_iter))
            pbar_b = tqdm(total=len(risk_train_iter), desc='risk train batch {:2d}'.format(epoch), leave=False)
            predict_all = np.array([[]], dtype=int)
            labels_all = np.array([[]], dtype=int)
            for batch in risk_train_iter:
                model.zero_grad()
                loss,pre=model(
                    input_ids=batch[0].to(device),
                    attention_mask=batch[3].to(device),
                    labels=batch[1].to(device),
                    task='risk'
                )
                loss = torch.mean(loss)
                risk_loss+=loss
                loss.backward()
                optimizer_r.step()
                pbar_b.update(1)
                pbar_b.set_postfix(loss='{:.3f}'.format(loss.item()))
                predic = pre.data.argmax(axis=1)
                labels_all = np.append(labels_all, batch[1].cpu().numpy())
                predict_all = np.append(predict_all, predic.cpu().numpy())
            pbar_b.close()
            torch.cuda.empty_cache()
            f1 = f1_score(labels_all, predict_all,average=None)
            print("----risk train f1----:",f1)
            micro_risk_f1 = f1_score(labels_all, predict_all, average='micro')
            macro_risk_f1 = f1_score(labels_all, predict_all, average='macro')
            print(f"risk train Micro-F1: {micro_risk_f1}")
            print(f"risk train Macro-F1: {macro_risk_f1}")
            print('risk train_loss:',risk_loss/len(risk_train_iter))
            pbar_b = tqdm(total=len(sensitive_train_iter), desc='sensitive train batch {:2d}'.format(epoch), leave=False)
            predict_all = np.array([[]], dtype=int)
            labels_all = np.array([[]], dtype=int)
            for batch in sensitive_train_iter:
                model.zero_grad()
                loss,pre=model(
                    input_ids=batch[0].to(device),
                    attention_mask=batch[3].to(device),
                    labels=batch[1].to(device),
                    task='sensitive'
                )
                loss = torch.mean(loss)
                s_loss+=loss
                loss.backward()
                optimizer_r.step()
                pbar_b.update(1)
                pbar_b.set_postfix(loss='{:.3f}'.format(loss.item()))
                predic = pre.data.argmax(axis=1)
                labels_all = np.append(labels_all, batch[1].cpu().numpy())
                predict_all = np.append(predict_all, predic.cpu().numpy())
            pbar_b.close()
            torch.cuda.empty_cache()
            f1 = f1_score(labels_all, predict_all,average=None)
            print("----sensitive train f1----:",f1)
            micro_f1 = f1_score(labels_all, predict_all, average='micro')
            macro_sensitive_f1 = f1_score(labels_all, predict_all, average='macro')
            print(f"sensitive train Micro-F1: {micro_f1}")
            print(f"sensitive train Macro-F1: {macro_sensitive_f1}")
            print('sensitive train_loss:',s_loss/len(sensitive_train_iter))
            if epoch%1==0:
                with torch.no_grad(): 
                    iv_loss=0
                    mv_loss=0
                    rv_loss=0
                    riv_loss=0
                    sv_loss=0
                    pbar_bv = tqdm(total=len(important_dev_iter), desc='important dev {:2d}'.format(epoch), leave=False)
                    predict_all = np.array([[]], dtype=int)
                    labels_all = np.array([[]], dtype=int)
                    for batch in important_dev_iter:
                        loss,pre=model(
                            input_ids=batch[0].to(device),
                            attention_mask=batch[3].to(device),
                            labels=batch[1].to(device),
                            task='important'
                        )
                        loss = torch.mean(loss)
                        iv_loss+=loss
                        predic = pre.data.argmax(axis=1)
                        labels_all = np.append(labels_all, batch[1].cpu().numpy())
                        predict_all = np.append(predict_all, predic.cpu().numpy())
                        f1 = f1_score(np.array(batch[1].clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
                        pbar_bv.update(1)
                        pbar_bv.set_postfix(loss='{:.3f}'.format(loss.item()))
                    f1 = f1_score(labels_all, predict_all,average=None)
                    print("----important val f1----:",f1)
                    micro_important_f1 = f1_score(labels_all, predict_all, average='micro')
                    macro_important_f1 = f1_score(labels_all, predict_all, average='macro')
                    print(f"important val Micro-F1: {micro_important_f1}")
                    print(f"important val Macro-F1: {macro_important_f1}")
                    pbar_bv.close()
                    torch.cuda.empty_cache()
                    print('important val_loss:',iv_loss/len(important_dev_iter))
                    pbar_mv = tqdm(total=len(multi_dev_iter), desc='multi dev {:2d}'.format(epoch), leave=False)
                    predict_all = np.array([[]], dtype=int)
                    labels_all = np.array([[]], dtype=int)
                    for batch in multi_dev_iter:
                        loss,pre =model(
                            input_ids=batch[0].to(device),
                            attention_mask=batch[3].to(device),
                            labels=batch[1].to(device),
                            task='multi'
                        )
                        loss = torch.mean(loss)
                        m_loss+=loss
                        m=torch.nn.Sigmoid()
                        predic = m(pre.data)
                        Threshold = 0.5
                        predic[predic > Threshold] = 1
                        predic[predic <= Threshold] = 0
                        if labels_all.shape[1] == 0:
                            labels_all=batch[1].cpu().numpy()
                            predict_all=predic.cpu().numpy()
                        else:
                            labels=batch[1].cpu().numpy()
                            predic = predic.cpu().numpy()
                            labels_all = np.vstack((labels_all, labels))
                            predict_all = np.vstack((predict_all, predic))
                        pbar_mv.update(1)
                        pbar_mv.set_postfix(loss='{:.3f}'.format(loss.item()))

                    f1 = f1_score(labels_all, predict_all,average=None)
                    print("----multi dev f1----:",f1)
                    micro_f1 = f1_score(labels_all, predict_all, average='micro')
                    macro_multi_f1 = f1_score(labels_all, predict_all, average='macro')
                    print(f"multi dev Micro-F1: {micro_f1}")
                    print(f"multi dev Macro-F1: {macro_multi_f1}")
                    pbar_mv.close()
                    torch.cuda.empty_cache()
                    print('multi dev_loss:',m_loss/len(multi_dev_iter))
                    pbar_bv = tqdm(total=len(risk_dev_iter), desc='risk dev {:2d}'.format(epoch), leave=False)
                    predict_all = np.array([[]], dtype=int)
                    labels_all = np.array([[]], dtype=int)
                    for batch in risk_dev_iter:
                        loss,pre=model(
                            input_ids=batch[0].to(device),
                            attention_mask=batch[3].to(device),
                            labels=batch[1].to(device),
                            task='risk'
                        )
                        loss = torch.mean(loss)
                        riv_loss+=loss
                        predic = pre.data.argmax(axis=1)
                        labels_all = np.append(labels_all, batch[1].cpu().numpy())
                        predict_all = np.append(predict_all, predic.cpu().numpy())
                        pbar_bv.update(1)
                        pbar_bv.set_postfix(loss='{:.3f}'.format(loss.item()))
                    f1 = f1_score(labels_all, predict_all,average=None)
                    print("----risk val f1----:",f1)
                    micro_risk_f1 = f1_score(labels_all, predict_all, average='micro')
                    macro_risk_f1 = f1_score(labels_all, predict_all, average='macro')
                    print(f"risk val Micro-F1: {micro_risk_f1}")
                    print(f"risk val Macro-F1: {macro_risk_f1}")
                    pbar_bv.close()
                    torch.cuda.empty_cache()
                    print('risk val_loss:',riv_loss/len(risk_dev_iter))
                    pbar_bv = tqdm(total=len(sensitive_dev_iter), desc='sensitive dev {:2d}'.format(epoch), leave=False)
                    predict_all = np.array([[]], dtype=int)
                    labels_all = np.array([[]], dtype=int)
                    for batch in sensitive_dev_iter:
                        loss,pre=model(
                            input_ids=batch[0].to(device),
                            attention_mask=batch[3].to(device),
                            labels=batch[1].to(device),
                            task='sensitive'
                        )
                        loss = torch.mean(loss)
                        sv_loss+=loss
                        predic = pre.data.argmax(axis=1)
                        labels_all = np.append(labels_all, batch[1].cpu().numpy())
                        predict_all = np.append(predict_all, predic.cpu().numpy())
                        pbar_bv.update(1)
                        pbar_bv.set_postfix(loss='{:.3f}'.format(loss.item()))
                    f1 = f1_score(labels_all, predict_all,average=None)
                    print("----sensitive val f1----:",f1)
                    micro_f1 = f1_score(labels_all, predict_all, average='micro')
                    macro_sensitive_f1 = f1_score(labels_all, predict_all, average='macro')
                    print(f"sensitive val Micro-F1: {micro_f1}")
                    print(f"sensitive val Macro-F1: {macro_sensitive_f1}")
                    pbar_bv.close()
                    torch.cuda.empty_cache()
                    print('sensitive val_loss:',sv_loss/len(sensitive_dev_iter))
            model.train()
            mean_f1=np.mean([macro_multi_f1,macro_risk_f1,macro_important_f1,macro_sensitive_f1])
            if mean_f1>max_dev_f1:
                torch.save(model.state_dict(), 'multi_xlnet2gpt_ddp2_cw.ckpt')
                print(f'epoch{epoch}分类f1性能提升:',mean_f1)
                max_dev_f1=mean_f1
    if rewrite:        
        model.load_state_dict(torch.load('multi_xlnet2gpt_ddp2_cw.ckpt'))
        for epoch in tqdm(range(100), desc="Epochs"):  # 例如，训练3个epoch
            r_loss=0

            pbar_r = tqdm(total=len(rewrite_train_dataloader), desc='rewrite batch {:2d}'.format(epoch), leave=False)
            for batch in rewrite_train_dataloader:
                model.zero_grad()
                loss=model(
                    input_ids=batch["source_ids"].to(device),
                    attention_mask=batch["source_mask"].to(device),
                    labels=batch['labels'].to(device),
                    decoder_input_ids=batch["target_ids"].to(device),
                    decoder_attention_mask=batch['target_mask'].to(device),
                    task='rewrite'
                )
                loss = torch.mean(loss)
                r_loss+=loss
                loss.backward()
                optimizer_rw.step()
                pbar_r.update(1)
                pbar_r.set_postfix(loss='{:.3f}'.format(loss.item()))
                pbar_r.close()
            torch.cuda.empty_cache()
            mean_loss=r_loss/len(rewrite_train_dataloader)
            perplexity = torch.exp(mean_loss).item()
            print(f"rewrite train Perplexity: {perplexity}")
            print('rewrite train_loss:',mean_loss)
            if epoch%1==0:
                rv_loss=0
                with torch.no_grad(): 
                    pbar_rv = tqdm(total=len(rewrite_val_dataloader), desc='rewrite batch {:2d}'.format(epoch), leave=False)
                    for batch in rewrite_val_dataloader:
                        loss =model(
                            input_ids=batch["source_ids"].to(device),
                            attention_mask=batch["source_mask"].to(device),
                            labels=batch['labels'].to(device),
                            decoder_input_ids=batch["target_ids"].to(device),
                            decoder_attention_mask=batch['target_mask'].to(device),
                            task='rewrite'
                        )
                        loss = torch.mean(loss)
                        rv_loss+=loss
                        pbar_rv.update(1)
                        pbar_rv.set_postfix(loss='{:.3f}'.format(loss.item()))
                    pbar_rv.close()
                    torch.cuda.empty_cache()
                    mean_loss=rv_loss/len(rewrite_val_dataloader)
                    perplexity = torch.exp(mean_loss).item()
                    print(f"rewrite dev Perplexity: {perplexity}")
                    print('rewrite dev_loss:',mean_loss)
            if  mean_loss<min_dev_loss:
                torch.save(model.state_dict(), 'multi_xlnet2gpt_ddp2_rw.ckpt')
                print(f'epoch{epoch}改写loss下降:',mean_loss)
                min_dev_loss=mean_loss


if test_step:
    if test_rewrite:
        model.load_state_dict(torch.load('multi_xlnet2gpt_ddp2_rw.ckpt'))
        model.eval()
        with torch.no_grad():
            with open(save_rewrite_path, 'w', encoding='utf-8') as fp:
                for text in true_false_adjective_tuples_test:
                    test_tokenized = encoder_tokenizer.encode_plus('summarization'+text[0][:250], return_tensors="pt")
                    test_input_ids = test_tokenized["input_ids"].to(device)
                    test_attention_mask = test_tokenized["attention_mask"].to(device)
                    if torch.torch.cuda.device_count() > 1:
                        beam_outputs = model.module.model.generate(
                                input_ids=test_input_ids,
                                attention_mask=test_attention_mask,
                                max_length=150,
                                early_stopping=True,
                                num_beams=10,
                                num_return_sequences=1,
                                no_repeat_ngram_size=2,
                            )
                    else:
                        beam_outputs = model.model.generate(
                                input_ids=test_input_ids,
                                attention_mask=test_attention_mask,
                                max_length=150,
                                early_stopping=True,
                                num_beams=10,
                                num_return_sequences=1,
                                no_repeat_ngram_size=2,
                            )
                    for beam_output in beam_outputs:
                        sent = decoder_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    sent=''.join(sent.split())
                    fp.write(json.dumps({'text': text[0], 'pred': sent, 'rewrite': text[1]},ensure_ascii=False) + '\n')
        print(rouge_scorces(save_rewrite_path))
    if test_classify:
            model.load_state_dict(torch.load('multi_xlnet2gpt_ddp2_cw.ckpt'))
            with torch.no_grad(): 

                iv_loss=0
                mv_loss=0
                riv_loss=0
                sv_loss=0
                pbar_bv = tqdm(total=len(important_test_iter), desc='important test {:2d}', leave=False)
                predict_all = np.array([[]], dtype=int)
                labels_all = np.array([[]], dtype=int)
                for batch in important_test_iter:
                    loss,pre=model(
                        input_ids=batch[0].to(device),
                        attention_mask=batch[3].to(device),
                        labels=batch[1].to(device),
                        task='important'
                    )
                    loss = torch.mean(loss)
                    iv_loss+=loss
                    predic = pre.data.argmax(axis=1)
                    labels_all = np.append(labels_all, batch[1].cpu().numpy())
                    predict_all = np.append(predict_all, predic.cpu().numpy())
                    f1 = f1_score(np.array(batch[1].clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
                    pbar_bv.update(1)
                    pbar_bv.set_postfix(loss='{:.3f}'.format(loss.item()))
                f1 = f1_score(labels_all, predict_all,average=None)
                print("----important test f1----:",f1)
                micro_important_f1 = f1_score(labels_all, predict_all, average='micro')
                macro_important_f1 = f1_score(labels_all, predict_all, average='macro')
                print(f"important test Micro-F1: {micro_important_f1}")
                print(f"important test Macro-F1: {macro_important_f1}")
                pbar_bv.close()
                torch.cuda.empty_cache()
                print('important test_loss:',iv_loss/len(important_test_iter))
                pbar_mv = tqdm(total=len(multi_test_iter), desc='multi test {:2d}', leave=False)
                predict_all = np.array([[]], dtype=int)
                labels_all = np.array([[]], dtype=int)
                for batch in multi_test_iter:
                    loss,pre =model(
                        input_ids=batch[0].to(device),
                        attention_mask=batch[3].to(device),
                        labels=batch[1].to(device),
                        task='multi'
                    )
                    loss = torch.mean(loss)
                    mv_loss+=loss
                    m=torch.nn.Sigmoid()
                    predic = m(pre.data)
                    Threshold = 0.5
                    predic[predic > Threshold] = 1
                    predic[predic <= Threshold] = 0
                    if labels_all.shape[1] == 0:
                        labels_all=batch[1].cpu().numpy()
                        predict_all=predic.cpu().numpy()
                    else:
                        labels=batch[1].cpu().numpy()
                        predic = predic.cpu().numpy()
                        labels_all = np.vstack((labels_all, labels))
                        predict_all = np.vstack((predict_all, predic))
                    pbar_mv.update(1)
                    pbar_mv.set_postfix(loss='{:.3f}'.format(loss.item()))

                f1 = f1_score(labels_all, predict_all,average=None)
                print("----multi test f1----:",f1)
                micro_f1 = f1_score(labels_all, predict_all, average='micro')
                macro_multi_f1 = f1_score(labels_all, predict_all, average='macro')
                report = metrics.classification_report(labels_all, predict_all, digits=6)
                print(f"multi test report: {report}")
                print(f"multi test Micro-F1: {micro_f1}")
                print(f"multi test Macro-F1: {macro_multi_f1}")
                pbar_mv.close()
                torch.cuda.empty_cache()
                print('multi test_loss:',mv_loss/len(multi_test_iter))
                pbar_bv = tqdm(total=len(risk_test_iter), desc='risk test {:2d}', leave=False)
                predict_all = np.array([[]], dtype=int)
                labels_all = np.array([[]], dtype=int)
                for batch in risk_test_iter:
                    loss,pre=model(
                        input_ids=batch[0].to(device),
                        attention_mask=batch[3].to(device),
                        labels=batch[1].to(device),
                        task='risk'
                    )
                    loss = torch.mean(loss)
                    riv_loss+=loss
                    predic = pre.data.argmax(axis=1)
                    labels_all = np.append(labels_all, batch[1].cpu().numpy())
                    predict_all = np.append(predict_all, predic.cpu().numpy())
                    pbar_bv.update(1)
                    pbar_bv.set_postfix(loss='{:.3f}'.format(loss.item()))
                f1 = f1_score(labels_all, predict_all,average=None)
                print("----risk test f1----:",f1)
                micro_risk_f1 = f1_score(labels_all, predict_all, average='micro')
                macro_risk_f1 = f1_score(labels_all, predict_all, average='macro')
                print(f"risk test Micro-F1: {micro_risk_f1}")
                print(f"risk test Macro-F1: {macro_risk_f1}")
                pbar_bv.close()
                torch.cuda.empty_cache()
                print('risk test_loss:',riv_loss/len(risk_test_iter))
                pbar_bv = tqdm(total=len(sensitive_test_iter), desc='sensitive dev {:2d}', leave=False)
                predict_all = np.array([[]], dtype=int)
                labels_all = np.array([[]], dtype=int)
                for batch in sensitive_test_iter:
                    loss,pre=model(
                        input_ids=batch[0].to(device),
                        attention_mask=batch[3].to(device),
                        labels=batch[1].to(device),
                        task='sensitive'
                    )
                    loss = torch.mean(loss)
                    sv_loss+=loss
                    predic = pre.data.argmax(axis=1)
                    labels_all = np.append(labels_all, batch[1].cpu().numpy())
                    predict_all = np.append(predict_all, predic.cpu().numpy())
                    pbar_bv.update(1)
                    pbar_bv.set_postfix(loss='{:.3f}'.format(loss.item()))
                f1 = f1_score(labels_all, predict_all,average=None)
                print("----sensitive test f1----:",f1)
                micro_f1 = f1_score(labels_all, predict_all, average='micro')
                macro_sensitive_f1 = f1_score(labels_all, predict_all, average='macro')
                print(f"sensitive test Micro-F1: {micro_f1}")
                print(f"sensitive test Macro-F1: {macro_sensitive_f1}")
                pbar_bv.close()
                torch.cuda.empty_cache()
                print('sensitive test_loss:',sv_loss/len(sensitive_test_iter))
