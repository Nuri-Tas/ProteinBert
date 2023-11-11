

PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert' # 'Rostlab/prot_bert_bfd_localization'
MAX_LEN = 512
SEQ_SIZE_LIMIT = 1000
BATCH_SIZE = 1
N_EPOCHS = 10

import re
import time
import os
import sys
import json
import pandas as pd
from tqdm import tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_optimizer.lamb import Lamb

from transformers import BertModel, BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, multilabel_confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pl.seed_everything(RANDOM_SEED)

def prepare_dataset(datapath, term_sets=('mf',), seq_size_limit=SEQ_SIZE_LIMIT):
	df = pd.read_excel(datapath)
	df = df[df['Seq'].apply(len) <= seq_size_limit]
	df['Seq'] = df['Seq'].apply(lambda x: ' '.join(list(re.sub(r"[UZOB]", "X", x))))
	meta = json.load(open(datapath.replace('chain.xlsx', 'meta')))
	all_terms = []
	present_df_list = []
	for term_set in term_sets:
		terms = list(meta[term_set].keys())
		for term in terms:
			term_presence_df = df[df['is_train'] & df["MF"].notna()]["MF"].apply(lambda x: int(term in x))
			if term_presence_df.sum() > 0:
				new_present_df = df[term_set.upper()].apply(lambda x: int(term in x) if type(x) == str else 0)
				present_df_list.append(new_present_df)
				all_terms.append(term)
	final_present_df = pd.concat(present_df_list, axis=1)
	final_present_df.columns = all_terms	
	df = pd.concat([df, final_present_df], axis=1)
	num_classes = len(all_terms)

	df = df.drop(columns=['ChainID', 'MF', 'BP', 'CC'])

	if 'is_test' in df.columns.tolist():
		train_df = df[df['is_train']].reset_index(drop=True).drop(columns=['is_train', 'is_valid', 'is_test'])
		valid_df = df[df['is_valid']].reset_index(drop=True).drop(columns=['is_train', 'is_valid', 'is_test'])
		test_df = df[df['is_test']].reset_index(drop=True).drop(columns=['is_train', 'is_valid', 'is_test'])
	else:
		train_df = df[df['is_train']].reset_index(drop=True).drop(columns=['is_train', 'is_valid'])
		valid_df = df[df['is_valid']].reset_index(drop=True).drop(columns=['is_train', 'is_valid'])
		test_df = valid_df.copy()

	weights = torch.FloatTensor((train_df.iloc[:, 1:].sum().sum() - train_df.iloc[:, 1:].sum()) / train_df.iloc[:, 1:].sum())
	return train_df, valid_df, test_df, num_classes, all_terms, weights

pdb_chain_path = "pdb_chain.xlsx"
train_df, valid_df, test_df, num_classes, LABEL_COLUMNS, weights = prepare_dataset(pdb_chain_path, term_sets=['mf'])
# train_df, valid_df, test_df, num_classes, LABEL_COLUMNS = prepare_dataset('swiss_chain.xlsx', term_sets=['bp'])
# train_df, valid_df, test_df, num_classes, LABEL_COLUMNS = prepare_dataset('swiss_chain.xlsx', term_sets=['cc'])
# train_df, valid_df, test_df, num_classes, LABEL_COLUMNS = prepare_dataset('swiss_chain.xlsx', term_sets=['mf', 'bp', 'cc'])

class ProteinSequenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.label_cols = self.df.columns[1:]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        sequence = self.df.iloc[item]['Seq']
        labels = self.df.iloc[item][self.label_cols].values.tolist()
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.FloatTensor(labels)
        }

class GOTermsDataModule(pl.LightningDataModule):
	def __init__(self, train_df, val_df, test_df, tokenizer, batch_size=8, max_token_len=128):
		super().__init__()
		self.batch_size = batch_size
		self.train_df = train_df
		self.val_df = val_df
		self.test_df = test_df
		self.tokenizer = tokenizer
		self.max_token_len = max_token_len

	def setup(self, stage=None):
		self.train_dataset = ProteinSequenceDataset(
			df=self.train_df,
			tokenizer=self.tokenizer,
			max_len=self.max_token_len
		)

		self.val_dataset = ProteinSequenceDataset(
			df=self.val_df,
			tokenizer=self.tokenizer,
			max_len=self.max_token_len
		)

		self.test_dataset = ProteinSequenceDataset(
			df=self.test_df,
			tokenizer=self.tokenizer,
			max_len=self.max_token_len
		)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=0
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			num_workers=0
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			num_workers=0
		)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if pos_weight is not None:
          self.pos_weight = pos_weight.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
          self.pos_weight = pos_weight
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class GOTermTagger(pl.LightningModule):
	def __init__(self, n_classes, weights, n_training_steps=None, n_warmup_steps=None):
		super().__init__()
		self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
		self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(self.bert.config.hidden_size, n_classes))
		self.num_classes = n_classes
		self.n_training_steps = n_training_steps
		self.n_warmup_steps = n_warmup_steps
		self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
		#self.criterion = nn.BCEWithLogitsLoss()
		#self.criterion = FocalLoss(gamma=2.25, pos_weight=weights)
		#self.criterion = FocalLoss(gamma=2.25)

	def forward(self, input_ids, attention_mask, labels=None):
		output = self.bert(input_ids, attention_mask=attention_mask)
		#output = self.classifier(output[0][0, 0, :].view(-1, self.bert.config.hidden_size))
		output = self.classifier(output.pooler_output) # ana olarak bu
		loss = 0 if labels is None else self.criterion(output, labels)
		return loss, output

	def training_step(self, batch, batch_idx):
		input_ids = batch["input_ids"]
		attention_mask = batch["attention_mask"]
		labels = batch["labels"]
		loss, outputs = self(input_ids, attention_mask, labels)
		self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
		return {"loss": loss, "predictions": outputs.detach(), "labels": labels.detach()}

	def validation_step(self, batch, batch_idx):
		input_ids = batch["input_ids"]
		attention_mask = batch["attention_mask"]
		labels = batch["labels"]
		loss, outputs = self(input_ids, attention_mask, labels)
		self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
		return loss

	def test_step(self, batch, batch_idx):
		input_ids = batch["input_ids"]
		attention_mask = batch["attention_mask"]
		labels = batch["labels"]
		loss, outputs = self(input_ids, attention_mask, labels)
		self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
		return loss

	def configure_optimizers(self):
		optimizer = AdamW(self.parameters(), lr=3e-6, eps=1e-8)
		# optimizer = Lamb(self.parameters(), lr=0.002, weight_decay=0.01)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.n_warmup_steps,
			num_training_steps=self.n_training_steps
		)

		return dict(
			optimizer=optimizer,
			lr_scheduler=dict(
				scheduler=scheduler,
				interval='step'
			)
		)

		return optimizer

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)

data_module = GOTermsDataModule(
	train_df,
	valid_df,
	test_df,
	tokenizer,
	batch_size=BATCH_SIZE,
	max_token_len=MAX_LEN
)

steps_per_epoch = train_df.shape[0] // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = steps_per_epoch

model = GOTermTagger(
	n_classes=num_classes,
	weights=weights,
	n_warmup_steps=warmup_steps,
	n_training_steps=total_training_steps
)

# for param in model.bert.parameters():
#     param.requires_grad = False

checkpoint_callback = ModelCheckpoint(
	dirpath="checkpoints",
	filename="best-checkpoint",
	save_top_k=1,
	verbose=True,
	monitor="val_loss",
	mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="go-terms")

early_stopping_callback = EarlyStopping(min_delta=1e-5, monitor='val_loss', patience=2)

# replace "gpus=1" parameter with these new two: "devices=1", "accelerator=gpu"
trainer = pl.Trainer(
	logger=logger,
	callbacks=[early_stopping_callback, checkpoint_callback],
	max_epochs=N_EPOCHS,
	val_check_interval=valid_df.shape[0] // BATCH_SIZE,
	devices=1,
	accelerator="cpu",
)

trainer.fit(model, data_module)
print('training done')
# test

trained_model = GOTermTagger.load_from_checkpoint(
	trainer.checkpoint_callback.best_model_path,
	n_classes=num_classes,
	weights=weights
)
trained_model.eval()
trained_model.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)

val_dataset = ProteinSequenceDataset(
	df=valid_df,
	tokenizer=tokenizer,
	max_len=MAX_LEN
)

predictions = []
labels = []
bert_embeddings = []

m = nn.Sigmoid()

for item in tqdm(val_dataset):
	prediction = trained_model(
		item["input_ids"].unsqueeze(dim=0).to(device),
		attention_mask=item["attention_mask"].unsqueeze(dim=0).to(device)
	)
	predictions.append(m(prediction[1].flatten()))
	labels.append(item["labels"].int())

predictions = torch.stack(predictions).detach().cpu()
labels = torch.stack(labels).detach().cpu()

y_pred = predictions.numpy()
y_true = labels.numpy()

upper, lower = 1, 0

y_pred2 = np.where(y_pred > 0.1, upper, lower)

print(classification_report(
	y_true,
	y_pred2,
	target_names=LABEL_COLUMNS,
))










### Embed all protein sequences

def infer_all_cls_outputs(df):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
	model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
	model.eval()

	dataset = ProteinSequenceDataset(
		df,
		tokenizer,
		SEQ_SIZE_LIMIT
	)

	tensors = []

	with torch.no_grad():
		for item in tqdm(dataset):
			output = model(
				item["input_ids"].unsqueeze(dim=0).to(device),
				attention_mask=item["attention_mask"].unsqueeze(dim=0).to(device)
			)

			tensors.append(output.pooler_output.detach().cpu())

	return tensors

train_tensors = infer_all_cls_outputs(train_df)
torch.save(torch.stack([i[0] for i in train_tensors]), '/tmp/train.pt')

