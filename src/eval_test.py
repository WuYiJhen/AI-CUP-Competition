import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertConfig, BertModel
from torchtext import data
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import chain

gpu = 0
device = torch.device('cuda:%d' % (
    gpu) if torch.cuda.is_available() else 'cpu')

####################################################################
###################### Load data & preprocess ######################
####################################################################

PRETRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

with open('./token_info/label2int.pickle', 'rb') as file:
    label2int = pickle.load(file)
file.close()


def remain_text(x):
    return str(x)


def bert_tokenize(x):
    x = tokenizer.tokenize(x)
    if len(x) > 511:
        return x[:256] + x[-255:]
    return x


def label_onehot(x):
    x = x.strip().split(' ')
    res = np.zeros(len(label2int))
    for k in x:
        if k in label2int:
            res[label2int[k]] = 1
    return res


CONTENT = data.Field(sequential=True, tokenize=bert_tokenize, preprocessing=tokenizer.convert_tokens_to_ids, init_token=tokenizer.cls_token_id,
                     pad_token=tokenizer.pad_token_id, unk_token=tokenizer.unk_token_id, use_vocab=False, lower=True, batch_first=True, include_lengths=False)
LABEL = data.LabelField(sequential=False, tokenize=remain_text,
                        preprocessing=label_onehot, dtype=torch.float, lower=False, use_vocab=False)
ID = data.LabelField(sequential=False, tokenize=remain_text,
                     dtype=torch.float, lower=False, use_vocab=False)

DataSet = data.TabularDataset(
    path='./data/test_data.csv', format='csv',
    fields={'content': ('content', CONTENT),
            'Id': ('Id', ID)}
)

BATCH_SIZE = 24
valid_iterator = data.BucketIterator(
    DataSet,
    batch_size=BATCH_SIZE,
    device=device,
    sort=False
)

int2label = {v: k for k, v in label2int.items()}


def inference(batch, model, device):
    model.eval()
    logits = model(batch)
    return logits


model_path = './model/bert-multilabel_best.pkl'
model = torch.load(model_path, map_location=device)
model.to(device)

threshold = 0.345
each_threshold = {'ENGINEERING': threshold,
                  'THEORETICAL': threshold,
                  'EMPIRICAL': threshold,
                  'OTHERS': threshold}

ids = []
predictions = []

len_valid = len(valid_iterator)
valid_loop = tqdm(enumerate(valid_iterator), total=len_valid, position=0)

for batch_idx, batch in valid_loop:
    logits = inference(batch, model, device)
    logits = nn.Sigmoid()(logits)
    logits = logits.detach().cpu().numpy()
    Id = batch.Id.detach().cpu().numpy()
    for i in range(len(logits)):
        predictions.append(list(logits[i]))
        ids.append(int(Id[i]))

tagging = []
for i in range(len(ids)):
    y_hat = [int2label[i] for i, x in enumerate(
        predictions[i]) if x >= each_threshold[int2label[i]]]
    most_pred = np.argmax(predictions[i])
    if len(y_hat) == 0:
        y_hat = [int2label[most_pred]]
    tagging.append(y_hat)

res_id = []
for i in range(len(ids)):
    res_id.append([ids[i]]*len(tagging[i]))

submit = pd.DataFrame(
    {'Id': list(chain(*res_id)), 'pred': list(chain(*tagging))})
submit = pd.get_dummies(submit, 'pred')
submit = submit.groupby('Id').sum().reset_index()
submit.columns = ['Id', 'EMPIRICAL', 'ENGINEERING', 'OTHERS', 'THEORETICAL']
submit = submit[['Id', 'THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']]

submit.to_csv('./data/submit.csv', index=False)
