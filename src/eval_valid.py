import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertConfig, BertModel
from torchtext import data
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

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

DataSet = data.TabularDataset(
    path='./data/valid_data.csv', format='csv',
    fields={'content': ('content', CONTENT),
            'label': ('label', LABEL)}
)

BATCH_SIZE = 16
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

docids = []
predictions = []
ground_truth = []

len_valid = len(valid_iterator)
valid_loop = tqdm(enumerate(valid_iterator), total=len_valid, position=0)

for batch_idx, batch in valid_loop:
    logits = inference(batch, model, device)
    logits = nn.Sigmoid()(logits)
    logits = logits.detach().cpu().numpy()
    y = batch.label.detach().cpu().numpy()
    for i in range(len(y)):
        tmp = []
        for j in np.where(y[i] == 1)[0]:
            tmp.append(int2label[j])
        tmp.sort()
        ground_truth.append(tmp)
    for i in range(len(logits)):
        predictions.append(list(logits[i]))

thresholds = np.arange(0.20, 0.45, 0.005)
thresholds = [round(x, 5) for x in thresholds]
true_label = {}
predict_label = {}
predict_true = {}
predict_false = {}
recall = {}
precision = {}
for threshold in thresholds:
    predict_label[threshold] = {}
    predict_true[threshold] = {}
    predict_false[threshold] = {}
    recall[threshold] = {}
    precision[threshold] = {}


for i in tqdm(range(len(ground_truth))):
    y = ground_truth[i]
    for k in y:
        if k not in true_label:
            true_label[k] = 0
        true_label[k] += 1
    for threshold in thresholds:
        y_hat = [int2label[i]
                 for i, x in enumerate(predictions[i]) if x >= threshold]
        most_pred = np.argmax(predictions[i])
        if len(y_hat) == 0:
            y_hat = [int2label[most_pred]]
        y_hat.sort()
        for k in y_hat:
            if k not in predict_label[threshold]:
                predict_label[threshold][k] = 0
            predict_label[threshold][k] += 1
            if k in y:
                if k not in predict_true[threshold]:
                    predict_true[threshold][k] = 0
                predict_true[threshold][k] += 1
            else:
                if k not in predict_false[threshold]:
                    predict_false[threshold][k] = 0
                predict_false[threshold][k] += 1


for threshold in thresholds:
    for k in true_label:
        if k in predict_true[threshold]:
            recall[threshold][k] = predict_true[threshold][k]/true_label[k]
        else:
            recall[threshold][k] = 0
    for k in predict_label[threshold]:
        if k in predict_true[threshold]:
            precision[threshold][k] = predict_true[threshold][k] / \
                predict_label[threshold][k]
        else:
            precision[threshold][k] = 0

micro_f1 = {}
performance = pd.DataFrame(
    columns=['threshold', 'recall', 'precision', 'f1-score(micro)'])
idx = 0
for threshold in thresholds:
    df_label = list(true_label.keys())
    df_true_count = [true_label[x] for x in df_label]
    df_true_predict = [predict_true[threshold][x]
                       if x in predict_true[threshold] else 0 for x in df_label]
    df_predict_count = [predict_label[threshold][x]
                        if x in predict_label[threshold] else 0 for x in df_label]
    df_recall = [round(recall[threshold][x], 3) for x in df_label]
    df_precision = [round(precision[threshold][x], 3)
                    if x in precision[threshold] else 0 for x in df_label]
    df = pd.DataFrame({'label': df_label, 'true_count': df_true_count, 'predict_count': df_predict_count,
                       'correct': df_true_predict, 'recall': df_recall, 'precision': df_precision})
    df = df.sort_values(
        by='true_count', ascending=False).reset_index(drop=True)
    mi_pre = sum(df['correct'])/sum(df['predict_count'])
    mi_rec = sum(df['correct'])/sum(df['true_count'])
    micro_f1[threshold] = 2 * (mi_pre*mi_rec)/(mi_pre+mi_rec)

    performance.loc[idx] = [threshold, round(
        mi_rec*100, 2), round(mi_pre*100, 2), round(micro_f1[threshold]*100, 2)]
    idx += 1
performance = performance.sort_values(
    by='f1-score(micro)', ascending=False).reset_index(drop=True)

performance.to_csv('./data/performance_valid_data.csv', index=False)
