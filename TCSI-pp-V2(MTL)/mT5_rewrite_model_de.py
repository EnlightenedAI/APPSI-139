import torch
import json
import argparse
import os
import copy
from Data_loading import dataload
from torch.utils.data import Dataset, DataLoader
from eval import rouge_scorces
import pytorch_lightning as pl
from loss.focallooss import Focal_Loss,Focal_Loss_multi
binary_loss=Focal_Loss(alpha=0.4, gamma=2) 
multi_loss=Focal_Loss_multi(alpha=0.4, gamma=2) 
loss_fct=torch.nn.CrossEntropyLoss(ignore_index=-100)
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    MT5Config,
)
from torch.optim import AdamW
pl.seed_everything(42)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(model_name)
t5_model = MT5ForConditionalGeneration.from_pretrained(model_name)
t5_config = MT5Config.from_pretrained(model_name)
t5_model=t5_model.to(device)
t5_model.config.eos_token_id = tokenizer.eos_token_id
t5_model.config.pad_token_id = tokenizer.pad_token_id

batch_size=16
max_len_inp=250
max_len_out=150
max_epochs=100
train_is_ture=False
testdata_is_ture=True
lr=3e-4
data_path=f'TCSI_pp/preprocessing/results_rewrite/'
savepath=f"./result/mt5_rewrite_de_epoch{max_epochs}.ckpt"
save_rewrite_path=f'./result/mt5_rewrite_de_epoch{max_epochs}.json'
true_false_adjective_tuples_train,true_false_adjective_tuples_validation,true_false_adjective_tuples_test=dataload(data_path)

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
config = MT5Config.from_pretrained(model_name)
config.pad_size=150
config.tokenizer=tokenizer
model = CustomMT5Model(config,t5_model)
model.load_state_dict(torch.load('multi_mt5_ddp2_cw1.ckpt'))
t5_model=model.model



class FalseGenerationDataset(Dataset):
    def __init__(self, tokenizer, tf_list, max_len_inp=250, max_len_out=150):
        self.true_false_adjective_tuples = tf_list
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
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
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100
        return {"source_ids": source_ids, "source_mask": src_mask,  "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels}

    def _build(self):
        for inputs, outputs in self.true_false_adjective_tuples:
            input_sent = "summarization: " + inputs[:400]
            ouput_sent = outputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True,return_tensors="pt"
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True,return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

train_dataset = FalseGenerationDataset(tokenizer,true_false_adjective_tuples_train, max_len_inp, max_len_out)
validation_dataset = FalseGenerationDataset(tokenizer,true_false_adjective_tuples_validation, max_len_inp, max_len_out)

class FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(FineTuner, self).__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        self.model = t5_model
        self.tokenizer = tokenizer

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
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.hparams.batch_size, num_workers=0)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        decay = ['decoder', 'lm_head']
        decoder_params = [p for n, p in param_optimizer if any(nd in n for nd in decay)]
        optimizer = torch.optim.AdamW(decoder_params, lr=lr, weight_decay=1e-5)
        return optimizer


args_dict = dict(
    batch_size=batch_size,
)
args = argparse.Namespace(**args_dict)
print('args',args)
model = FineTuner(args).to(device)
if train_is_ture:
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, progress_bar_refresh_rate=1)
    print(trainer)
    trainer.fit(model)
    trainer.save_checkpoint(savepath)

if testdata_is_ture:
    new_model = FineTuner(args).to(device)
    new_model = new_model.load_from_checkpoint(checkpoint_path=savepath)
    new_model.to(device)
    new_model.eval()
    new_model.model.to(device)
    with open(save_rewrite_path, 'w', encoding='utf-8') as fp:
        for text in true_false_adjective_tuples_test:
            test_tokenized = tokenizer.encode_plus('summarization'+text[0][:350], return_tensors="pt")
            test_input_ids = test_tokenized["input_ids"].to(device)
            test_attention_mask = test_tokenized["attention_mask"].to(device)
            beam_outputs = new_model.model.generate(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                max_length=250,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )
            for beam_output in beam_outputs:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            fp.write(json.dumps({'text': text[0], 'pred': sent, 'rewrite': text[1]},ensure_ascii=False) + '\n')

print(rouge_scorces(save_rewrite_path))
# torch.cuda.empty_cache()