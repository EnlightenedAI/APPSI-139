import torch
import copy
import os
import argparse
import json
from eval import rouge_scorces
from Data_loading import dataload
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    BertTokenizerFast,
    EncoderDecoderModel,
    BertTokenizer,
    AutoTokenizer,
    GPT2Tokenizer
)
from torch.optim import AdamW
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
pl.seed_everything(42)
bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
bert_tokenizer.bos_token = bert_tokenizer.cls_token
bert_tokenizer.eos_token = bert_tokenizer.sep_token
gpt2_tokenizer =GPT2Tokenizer.from_pretrained("openai-community/gpt2")
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_tokenizer.bos_token = gpt2_tokenizer.cls_token
gpt2_tokenizer.eos_token = gpt2_tokenizer.sep_token
roberta2gpt_model = EncoderDecoderModel.from_encoder_decoder_pretrained("xlm-roberta-base", "openai-community/gpt2")
tokens = ["[SEP]"]
bert_tokenizer.add_tokens(tokens, special_tokens=True)
roberta2gpt_model=roberta2gpt_model.to('cuda')
roberta2gpt_model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
roberta2gpt_model.config.eos_token_id = gpt2_tokenizer.eos_token_id
roberta2gpt_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
roberta2gpt_model.config.vocab_size = roberta2gpt_model.config.decoder.vocab_size

max_len_inp=250
max_len_out=150
train_is_ture=True
num_epoch=100
source_is_ture=True
testdata_is_ture=True
batch_size=32
data_path=f'CAPP_130_Corpus/Subdataset/Rewrite_Sentences/'
savepath=f"./result_multi/roberta2bert_rewrite_epoch{num_epoch}.ckpt" #save model
save_rewrite_path=f'./result_multi/roberta2bert_rewrite_epoch{num_epoch}.json' #save rewrite result
lr=3e-4
true_false_adjective_tuples_train,true_false_adjective_tuples_validation,true_false_adjective_tuples_test=dataload(data_path,only_test=False)#

class FalseGenerationDataset(Dataset):
    def __init__(self, bert_tokenizer,gpt_tokenizer, tf_list, max_len_inp=250, max_len_out=150,templates_is_ture=False):
        self.true_false_adjective_tuples = tf_list
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.bert_tokenizer = bert_tokenizer #token-model
        self.gpt_tokenizer = gpt_tokenizer
        self.inputs = []
        self.templates=[]
        self.targets = []
        self.skippedcount = 0
        self.templates_is_ture = templates_is_ture
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
        for inputs,outputs in self.true_false_adjective_tuples:
            input_sent =  "summarization: " + inputs[:350]
            ouput_sent = outputs
            tokenized_inputs = self.bert_tokenizer.batch_encode_plus(
                [input_sent], padding='max_length',max_length=self.max_len_input,truncation=True, return_tensors="pt"
            )
            tokenized_targets = self.gpt_tokenizer.batch_encode_plus(
                [ouput_sent],  padding='max_length',max_length=self.max_len_output,truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

train_dataset = FalseGenerationDataset(bert_tokenizer,gpt2_tokenizer,true_false_adjective_tuples_train, max_len_inp, max_len_out)
validation_dataset = FalseGenerationDataset(bert_tokenizer,gpt2_tokenizer,true_false_adjective_tuples_validation, max_len_inp, max_len_out)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        self.model = roberta2gpt_model
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
model = T5FineTuner(args).to('cuda')
if train_is_ture:
    trainer = pl.Trainer(max_epochs = num_epoch,amp_level='01', gpus=1,progress_bar_refresh_rate=30)
    trainer.fit(model)
    trainer.save_checkpoint(savepath)

if testdata_is_ture:
    tokenizer = bert_tokenizer
    new_model = T5FineTuner(args).to('cuda')
    savepath = 'roberta2gpt_rewrite_epoch100.ckpt'
    new_model = new_model.load_from_checkpoint(checkpoint_path=savepath)
    trainer = pl.Trainer(gpus=1)
    new_model.cuda()
    new_model.eval()
    new_model.model.cuda()
    with open(save_rewrite_path, 'w', encoding='utf-8') as fp:
        for text in true_false_adjective_tuples_test:
            test_tokenized = tokenizer.encode_plus('summarization'+text[0][:350], return_tensors="pt")
            test_input_ids = test_tokenized["input_ids"].to('cuda')
            test_attention_mask = test_tokenized["attention_mask"].to('cuda')
            beam_outputs = new_model.model.generate(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                max_length=150,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=30,
            )
            for beam_output in beam_outputs:
                sent = gpt2_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            fp.write(json.dumps({'text': text[0],'pred':sent,'rewrite':text[1]}, ensure_ascii=False) + '\n')
if source_is_ture:
    print(rouge_scorces(save_rewrite_path))
