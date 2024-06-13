import torch
import os
import copy
import argparse
import json
from Data_loading import dataload
from eval import rouge_scorces
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    # AdamW,
    BertTokenizerFast,
    EncoderDecoderModel,
    AutoTokenizer,
    GPT2Config,
    GPT2TokenizerFast
)
from torch.optim import AdamW
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"]='1'
from loss.focallooss import Focal_Loss,Focal_Loss_multi
# 检查是否安装了必要的库
binary_loss=Focal_Loss(alpha=0.4, gamma=2) 
multi_loss=Focal_Loss_multi(alpha=0.4, gamma=2) 
# loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
loss_fct=torch.nn.CrossEntropyLoss(ignore_index=-100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
pl.seed_everything(42)
from transformers import EncoderDecoderModel, BertTokenizer, GPT2Tokenizer

# 加载BERT和GPT的Tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained('xlnet/xlnet-base-cased')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
xlnet2gpt_model = EncoderDecoderModel.from_encoder_decoder_pretrained('xlnet/xlnet-base-cased', "openai-community/gpt2")
# bert2gpt_model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "openai-community/gpt2")
xlnet2gpt_model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
xlnet2gpt_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
# bert_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-cased")
# # bert_tokenizer.bos_token = bert_tokenizer.cls_token
# # bert_tokenizer.eos_token = bert_tokenizer.sep_token
# gpt2_tokenizer =GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
# # gpt2_tokenizer.bos_token = gpt2_tokenizer.cls_token
# # gpt2_tokenizer.eos_token = gpt2_tokenizer.sep_token
# # gpt2_tokenizer.decoder_start_token_id= gpt2_tokenizer.bos_token_id
# print("_" * 10, 'loading bert', "_" * 10)
# bert2gpt_model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "openai-community/gpt2")
# bert2gpt_model=bert2gpt_model.to(device)
# bert2gpt_model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
# bert2gpt_model.config.eos_token_id = gpt2_tokenizer.eos_token_id
# bert2gpt_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
# bert2gpt_model.config.vocab_size = bert2gpt_model.config.decoder.vocab_size
config = AutoTokenizer.from_pretrained('xlnet/xlnet-base-cased')
dconfig=GPT2Config.from_pretrained('openai-community/gpt2')
print(config ,dconfig )
config.pad_size=150
config.hidden_size=768
config.decoder_vocab_size=dconfig.vocab_size
config.tokenizer=bert_tokenizer
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
            if not labels is None:
                loss = binary_loss(classification_output, labels)
            # loss.backward()
                return loss,classification_output
            else:
                return classification_output
        elif task=='risk':
            # 二分类任务，使用编码器的输出
            # print(labels)
            # print(encoder_outputs.last_hidden_state.shape)
            classification_output = self.risk(encoder_outputs.last_hidden_state[:,0,:])
            # print(classification_output.shape)
            if not labels is None:
                loss = binary_loss(classification_output, labels)
            # loss.backward()
                return loss,classification_output
            else:
                return classification_output  
        elif task=='sensitive':
            # 二分类任务，使用编码器的输出
            # print(labels)
            # print(encoder_outputs.last_hidden_state.shape)
            classification_output = self.sensitive(encoder_outputs.last_hidden_state[:,0,:])
            # print(classification_output.shape)
            if not labels is None:
                loss = binary_loss(classification_output, labels)
            # loss.backward()
                return loss,classification_output
            else:
                return classification_output  
        elif task=='multi':
            # 多分类任务，使用编码器的输出
            # print(encoder_outputs.last_hidden_state.shape)
            classification_output = self.multi(encoder_outputs.last_hidden_state[:,0,:])
            # print(classification_output.shape)
            if not labels is None:
                loss = multi_loss(classification_output, labels)
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
            # # sequence_output = decoder_outputs[0]
            # # print(encoder_outputs.last_hidden_state.shape,sequence_output.shape)
            # # prediction_scores = self.model.decoder.lm_head(sequence_output)
            # # prediction_scores=prediction_scores.view(-1, self.config.vocab_size)
            # # print('预测：',prediction_scores.device,prediction_scores.shape)
            # prediction_scores=decoder_outputs.logits
            # loss = None
            # print(labels)
            # pdb.set_trace()
            # if labels is not None:
                
            #     loss = loss_fct(prediction_scores.view(-1, config.decoder_vocab_size), labels.view(-1))
            #     # print('loss：',loss.device,loss)
                
            output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            )
            loss=output[0]
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
# config.model=modelx
model = CustomMT5Model(config,xlnet2gpt_model)
# model = CustomMT5Model(config,t5_model)
model.load_state_dict(torch.load('multi_xlnet2gpt_ddp2_cw.ckpt'))
xlnet2gpt_model=model.model
# params = list(bert2gpt_model.parameters())
# print('model-size',len(params))

lr=1e-4

train_is_ture=True
num_epoch=50
source_is_ture=True
testdata_is_ture=True
batch_size=32
max_len_inp=150
max_len_out=150
data_path=f'TCSI_pp/preprocessing/results_rewrite/'
savepath=f"/root/autodl-tmp/xlnet2gpt_rewrite_epoch{num_epoch}.ckpt" #save model
save_rewrite_path=f'/root/autodl-tmp/xlnet2gpt_rewrite_epoch{num_epoch}.json' #save rewrite result

true_false_adjective_tuples_train,true_false_adjective_tuples_validation,true_false_adjective_tuples_test=dataload(data_path)

class FalseGenerationDataset(Dataset):
    def __init__(self, bert_tokenizer,gpt_tokenizer, data_list, max_len_inp=250, max_len_out=150):
        self.true_false_adjective_tuples = data_list
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.bert_tokenizer = bert_tokenizer
        self.gpt_tokenizer = gpt_tokenizer
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
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels}

    def _build(self):
        for inputs, outputs in self.true_false_adjective_tuples:
            input_sent = "summarization: " + inputs[:200]
            ouput_sent = outputs
            tokenized_inputs = self.bert_tokenizer.batch_encode_plus(
                [input_sent], padding='max_length', max_length=self.max_len_input, truncation=True, return_tensors="pt"
            )
            tokenized_targets = self.gpt_tokenizer.batch_encode_plus(
                [ouput_sent], padding='max_length', max_length=self.max_len_output, truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

train_dataset = FalseGenerationDataset(bert_tokenizer,gpt2_tokenizer,true_false_adjective_tuples_train, max_len_inp, max_len_out)
validation_dataset = FalseGenerationDataset(bert_tokenizer,gpt2_tokenizer,true_false_adjective_tuples_validation, max_len_inp, max_len_out)

class FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(FineTuner, self).__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        self.model = xlnet2gpt_model
        self.tokenizer = bert_tokenizer

    def forward(self, input_ids, attention_mask=None,template_ids=None,
            template_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                    lm_labels=None):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size,shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.hparams.batch_size, num_workers=0)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=lr, eps=1e-8)
        return optimizer

args_dict = dict(
    batch_size=batch_size,

)
args = argparse.Namespace(**args_dict)
model = FineTuner(args).to(device)

if train_is_ture:
    model = model.load_from_checkpoint(checkpoint_path='/root/autodl-tmp/xlnet2gpt_rewrite_epoch20.ckpt')
    
    # from pytorch_lightning.callbacks import ModelCheckpoint
    # dirpath = '/root/autodl-fs'
    # os.makedirs(dirpath, exist_ok=True)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=dirpath,
    #     filename='xlnet2gpt_ddp_rewrite',
    #     monitor='val_loss',
    #     mode='min',
    #     save_top_k=1,

    # )

    # trainer = pl.Trainer(callbacks=[checkpoint_callback],max_epochs=num_epoch, amp_level='01', progress_bar_refresh_rate=300,gpus=1)
   
    trainer = pl.Trainer(max_epochs = num_epoch,amp_level='01', gpus=1, progress_bar_refresh_rate=30)
    trainer.fit(model)
    trainer.save_checkpoint(savepath)

if testdata_is_ture:
    new_model = FineTuner(args).to(device)
    new_model = new_model.load_from_checkpoint(checkpoint_path=savepath)
    # trainer = pl.Trainer(gpus=1)
    new_model.to(device)
    new_model.eval()
    new_model.model.to(device)

    with open(save_rewrite_path, 'w', encoding='utf-8') as fp:
        for text in true_false_adjective_tuples_test:
            print(text)
            test_tokenized = bert_tokenizer.encode_plus("summarization: "+text[0][:200], return_tensors="pt")
            test_input_ids = test_tokenized["input_ids"].to(device)
            test_attention_mask = test_tokenized["attention_mask"].to(device)
            beam_outputs = new_model.model.generate(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                max_length=max_len_out,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
            )
            for beam_output in beam_outputs:
                sent = gpt2_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # sent=''.join(sent.split())
                # print(sent)
            fp.write(json.dumps({'text': text[0], 'pred': sent, 'rewrite': text[1]}, ensure_ascii=False) + '\n')

if source_is_ture:
    print(rouge_scorces(save_rewrite_path))